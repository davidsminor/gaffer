//////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2012-2013, John Haddon. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are
//  met:
//
//      * Redistributions of source code must retain the above
//        copyright notice, this list of conditions and the following
//        disclaimer.
//
//      * Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials provided with
//        the distribution.
//
//      * Neither the name of John Haddon nor the names of
//        any other contributors to this software may be used to endorse or
//        promote products derived from this software without specific prior
//        written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
//  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
//  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
//  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
//  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
//  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
//  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
//  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
//  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
//  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//////////////////////////////////////////////////////////////////////////

#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/pipeline.h"
#include "tbb/task.h"
#include "tbb/compat/thread"

#include "boost/lexical_cast.hpp"

#include "OpenEXR/ImathBoxAlgo.h"
#include "OpenEXR/ImathFun.h"

#include "IECore/AttributeBlock.h"
#include "IECore/MessageHandler.h"
#include "IECore/StateRenderable.h"
#include "IECore/AngleConversion.h"
#include "IECore/MotionBlock.h"
#include "IECore/SceneInterface.h"

#include "Gaffer/Context.h"
#include "Gaffer/ScriptNode.h"

#include "GafferScene/SceneProcedural.h"
#include "GafferScene/ScenePlug.h"

using namespace std;
using namespace Imath;
using namespace IECore;
using namespace Gaffer;
using namespace GafferScene;

int locationCount;

// TBB recommends that you defer decisions about how many threads to create
// to it, so you can write nice high level code and it can decide how best
// to schedule the work. Generally if left to do this, it schedules it by
// making as many threads as there are cores, to make best use of the hardware.
// This is all well and good, until you're running multiple renders side-by-side,
// telling the renderer to use a limited number of threads so they all play nicely 
// together. Let's use the example of a 32 core machine with 4 8-thread 3delight
// renders running side by side.
//
// - 3delight will make 8 threads. TBB didn't make them itself, so it considers
//   them to be "master" threads.
// - 3delight will then call our procedurals on some subset of those 8 threads.
//   We'll execute graphs, which may or may not use TBB internally, but even if they
//   don't, we're using parallel_for for child procedural construction.
// - TBB will be invoked from these master threads, see that it hasn't been
//   initialised yet, and merrily initialise itself to use 32 threads.
// - We now have 4 side by side renders each trying to take over the machine,
//   and a not-so-happy IT department.
//
// The "solution" to this is to explicitly initialise TBB every time a procedural
// is invoked, limiting it to a certain number of threads. Problem solved? Maybe.
// There's another wrinkle, in that TBB is initialised separately for each master
// thread, and if each master asks for a maximum of N threads, and there are M masters,
// TBB might actually make up to `M * N` threads, clamped at the number of cores.
// So with N set to 8, you could still get a single process trying to use the
// whole machine. In practice, it appears that 3delight perhaps doesn't make great
// use of procedural concurrency, so the worst case of M procedurals in flight,
// each trying to use N threads may not occur. What other renderers do in this
// situation is unknown.
//
// I strongly suspect that the long term solution to this is to abandon using
// a procedural hierarchy matching the scene hierarchy, and to do our own
// threaded traversal of the scene, outputting the results to the renderer via
// a single master thread. We could then be sure of our resource usage, and
// also get better performance with renderers unable to make best use of
// procedural concurrency.
//
// In the meantime, we introduce a hack. The GAFFERSCENE_SCENEPROCEDURAL_THREADS
// environment variable may be used to clamp the number of threads used by any
// given master thread. We sincerely hope to have a better solution before too
// long.
//
// Worthwhile reading :
//
// https://software.intel.com/en-us/blogs/2011/04/09/tbb-initialization-termination-and-resource-management-details-juicy-and-gory/
//
void initializeTaskScheduler( tbb::task_scheduler_init &tsi )
{
	assert( !tsi.is_active() );

	static int g_maxThreads = -1;
	if( g_maxThreads == -1 )
	{
		if( const char *c = getenv( "GAFFERSCENE_SCENEPROCEDURAL_THREADS" ) )
		{
			g_maxThreads = boost::lexical_cast<int>( c );
		}
		else
		{
			g_maxThreads = 0;
		}
	}

	if( g_maxThreads > 0 )
	{
		tsi.initialize( g_maxThreads );
	}
}

tbb::atomic<int> SceneProcedural::g_pendingSceneProcedurals;
tbb::mutex SceneProcedural::g_allRenderedMutex;

SceneProcedural::AllRenderedSignal SceneProcedural::g_allRenderedSignal;

SceneProcedural::SceneProcedural( ConstScenePlugPtr scenePlug, const Gaffer::Context *context, const ScenePlug::ScenePath &scenePath )
	:	m_scenePlug( scenePlug ), m_context( new Context( *context ) ), m_scenePath( scenePath ), m_rendered( false )
{
	tbb::task_scheduler_init tsi( tbb::task_scheduler_init::deferred );
	initializeTaskScheduler( tsi );

	// get a reference to the script node to prevent it being destroyed while we're doing a render:
	m_scriptNode = m_scenePlug->ancestor<ScriptNode>();

	m_context->set( ScenePlug::scenePathContextName, m_scenePath );

	// options

	Context::Scope scopedContext( m_context.get() );
	ConstCompoundObjectPtr globals = m_scenePlug->globalsPlug()->getValue();

	const BoolData *transformBlurData = globals->member<BoolData>( "option:render:transformBlur" );
	m_options.transformBlur = transformBlurData ? transformBlurData->readable() : false;

	const BoolData *deformationBlurData = globals->member<BoolData>( "option:render:deformationBlur" );
	m_options.deformationBlur = deformationBlurData ? deformationBlurData->readable() : false;

	const V2fData *shutterData = globals->member<V2fData>( "option:render:shutter" );
	m_options.shutter = shutterData ? shutterData->readable() : V2f( -0.25, 0.25 );
	m_options.shutter += V2f( m_context->getFrame() );

	// attributes

	transformBlurData = globals->member<BoolData>( "attribute:gaffer:transformBlur" );
	m_attributes.transformBlur = transformBlurData ? transformBlurData->readable() : true;
	
	const IntData *transformBlurSegmentsData = globals->member<IntData>( "attribute:gaffer:transformBlurSegments" );
	m_attributes.transformBlurSegments = transformBlurSegmentsData ? transformBlurSegmentsData->readable() : 1;
	
	deformationBlurData = globals->member<BoolData>( "attribute:gaffer:deformationBlur" );
	m_attributes.deformationBlur = deformationBlurData ? deformationBlurData->readable() : true;
	
	const IntData *deformationBlurSegmentsData = globals->member<IntData>( "attribute:gaffer:deformationBlurSegments" );
	m_attributes.deformationBlurSegments = deformationBlurSegmentsData ? deformationBlurSegmentsData->readable() : 1;

	computeBound();
	updateAttributes( true );
	++g_pendingSceneProcedurals;

}

SceneProcedural::SceneProcedural( const SceneProcedural &other, const ScenePlug::ScenePath &scenePath )
	:	m_scenePlug( other.m_scenePlug ), m_context( new Context( *(other.m_context), Context::Shared ) ), m_scenePath( scenePath ),
		m_options( other.m_options ), m_attributes( other.m_attributes ), m_rendered( false )
{
	tbb::task_scheduler_init tsi( tbb::task_scheduler_init::deferred );
	initializeTaskScheduler( tsi );

	// get a reference to the script node to prevent it being destroyed while we're doing a render:
	m_scriptNode = m_scenePlug->ancestor<ScriptNode>();

	m_context->set( ScenePlug::scenePathContextName, m_scenePath );

	computeBound();
	updateAttributes( false );
	++g_pendingSceneProcedurals;
}

SceneProcedural::~SceneProcedural()
{
	if( !m_rendered )
	{
		decrementPendingProcedurals();
	}
}

void SceneProcedural::computeBound()
{
	/// \todo I think we should be able to remove this exception handling in the future.
	/// Either when we do better error handling in ValuePlug computations, or when
	/// the bug in IECoreGL that caused the crashes in SceneProceduralTest.testComputationErrors
	/// is fixed.
	try
	{
		ContextPtr timeContext = new Context( *m_context, Context::Borrowed );
		Context::Scope scopedTimeContext( timeContext.get() );

		/// \todo This doesn't take account of the unfortunate fact that our children may have differing
		/// numbers of segments than ourselves. To get an accurate bound we would need to know the different sample
		/// times the children may be using and evaluate a bound at those times as well. We don't want to visit
		/// the children to find the sample times out though, because that defeats the entire point of deferred loading.
		///
		/// Here are some possible approaches :
		///
		/// 1) Add a new attribute called boundSegments, which defines the number of segments used to calculate
		///    the bounding box. It would be the responsibility of the user to set this to an appropriate value
		///    at the parent levels, so that the parents calculate bounds appropriate for the children.
		///    This seems like a bit too much burden on the user.
		///
		/// 2) Add a global option called "maxSegments" - this will clamp the number of segments used on anything
		///    and will be set to 1 by default. The user will need to increase it to allow the leaf level attributes
		///    to take effect, and all bounding boxes everywhere will be calculated using that number of segments
		///    (actually I think it'll be that number of segments and all nondivisible smaller numbers). This should
		///    be accurate but potentially slower, because we'll be doing the extra work everywhere rather than only
		///    where needed. It still places a burden on the user (increasing the global clamp appropriately),
		///    but not quite such a bad one as they don't have to figure anything out and only have one number to set.
		///
		/// 3) Have the StandardOptions node secretly compute a global "maxSegments" behind the scenes. This would
		///    work as for 2) but remove the burden from the user. However, it would mean preventing any expressions
		///    or connections being used on the segments attributes, because they could be used to cheat the system.
		///    It could potentially be faster than 2) because it wouldn't have to do all nondivisible numbers - it
		///    could know exactly which numbers of segments were in existence. It still suffers from the
		///    "pay the price everywhere" problem.

		std::set<float> times;
		motionTimes( ( m_options.deformationBlur && m_attributes.deformationBlur ) ? m_attributes.deformationBlurSegments : 0, times );
		motionTimes( ( m_options.transformBlur && m_attributes.transformBlur ) ? m_attributes.transformBlurSegments : 0, times );

		m_bound = Imath::Box3f();
		for( std::set<float>::const_iterator it = times.begin(), eIt = times.end(); it != eIt; it++ )
		{
			timeContext->setFrame( *it );
			Box3f b = m_scenePlug->boundPlug()->getValue();
			M44f t = m_scenePlug->transformPlug()->getValue();
			m_bound.extendBy( transform( b, t ) );
		}
	}
	catch( const std::exception &e )
	{
		m_bound = Imath::Box3f();
		IECore::msg( IECore::Msg::Error, "SceneProcedural::bound()", e.what() );
	}
}

Imath::Box3f SceneProcedural::bound() const
{
	return m_bound;
}

//////////////////////////////////////////////////////////////////////////
// SceneProceduralCreate implementation
//
// This uses tbb::parallel_for to fill up a preallocated array of
// SceneProceduralPtrs with new SceneProcedurals, based on the parent
// SceneProcedural and the child names we supply.
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::SceneProceduralCreate
{

	public:
		typedef std::vector<SceneProceduralPtr> SceneProceduralContainer;

		SceneProceduralCreate(
			SceneProceduralContainer &childProcedurals,
			const SceneProcedural &parent,
			const vector<InternedString> &childNames
			
		) :
			m_childProcedurals( childProcedurals ),
			m_parent( parent ),
			m_childNames( childNames )
		{
		}

		void operator()( const tbb::blocked_range<int> &range ) const
		{
			for( int i=range.begin(); i!=range.end(); ++i )
			{
				ScenePlug::ScenePath childScenePath = m_parent.m_scenePath;
				childScenePath.push_back( m_childNames[i] );
				SceneProceduralPtr sceneProcedural = new SceneProcedural( m_parent, childScenePath );
				m_childProcedurals[ i ] = sceneProcedural;
			}
		}
	
	private:
	
		SceneProceduralContainer &m_childProcedurals;
		const SceneProcedural &m_parent;
		const vector<InternedString> &m_childNames;
		
};



//////////////////////////////////////////////////////////////////////////
// SceneGraph implementation
//
// This is a node in a scene hierarchy, which gets built when the
// interactive render starts up. Each node stores the attribute hash of
// the input scene at its corresponding location in the hierarchy, so
// when the incoming scene updates, we are able to determine the locations
// at which the attributes have changed since the last update, reevaluate
// those attributes and send updates to the renderer.
//
// \todo: This is very similar to the SceneGraph mechanism in
// GafferSceneUI::SceneGadget. At some point it would be good to refactor
// this and use the same class for both of them. See the comments in
// src/GafferSceneUI/SceneGadget.cpp for details.
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::SceneGraph
{

	public :

		SceneGraph() :
			m_transformBlur( false ),
			m_transformBlurSegments( 0 ),
			m_deformationBlur( false ),
			m_deformationBlurSegments( 0 ),
			m_parent( NULL ),
			m_locationPresent( true )
		{
		}

		~SceneGraph()
		{
			for( std::vector<SceneGraph *>::const_iterator it = m_children.begin(), eIt = m_children.end(); it != eIt; ++it )
			{
				delete *it;
			}
		}
		
		void path( ScenePlug::ScenePath &p )
		{
			if( !m_parent )
			{
				return;
			}
			m_parent->path( p );
			p.push_back( m_name );
		}

		// motion blur info:
		bool m_transformBlur;
		int m_transformBlurSegments;
		
		bool m_deformationBlur;
		int m_deformationBlurSegments;
		
	private :
		
		friend class SceneGraphBuildTask;
		friend class ChildNamesUpdateTask;
		
		friend class SceneGraphIteratorFilter;
		friend class SceneGraphEvaluatorFilter;
		friend class SceneGraphOutputFilter;
		
		// scene structure data:
		IECore::InternedString m_name;
		SceneGraph *m_parent;
		std::vector<SceneProcedural::SceneGraph *> m_children;
		
		// hashes as of the most recent evaluation:
		IECore::MurmurHash m_attributesHash;
		IECore::MurmurHash m_childNamesHash;
		
		
		
		// actual scene data:
		IECore::ConstCompoundObjectPtr m_attributes;
		std::vector< std::pair< float, IECore::ConstObjectPtr > > m_objectSamples;
		std::vector< std::pair< float, Imath::M44f > > m_transformSamples;
		
		// flag indicating if this location is currently present - (used
		// when the child names change)
		bool m_locationPresent;
};



//////////////////////////////////////////////////////////////////////////
// BuildTask implementation
//
// We use this tbb::task to traverse the input scene and build the
// hierarchy for the first time. Recursion is terminated for locations
// at which scene:visible is set to false, so this task also computes
// and stores the attributes.
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::SceneGraphBuildTask : public tbb::task
{

	public :

		SceneGraphBuildTask( const ScenePlug *scene, const Context *context, SceneGraph *sceneGraph, const ScenePlug::ScenePath &scenePath )
			:	m_scene( scene ),
				m_context( context ),
				m_sceneGraph( sceneGraph ),
				m_scenePath( scenePath )
		{
		}

		~SceneGraphBuildTask()
		{
		}

		virtual task *execute()
		{
			ContextPtr context = new Context( *m_context, Context::Borrowed );
			context->set( ScenePlug::scenePathContextName, m_scenePath );
			Context::Scope scopedContext( context.get() );

			// we need the attributes so we can terminate recursion at invisible locations, so
			// we might as well store them in the scene graph, along with the hash:

			m_sceneGraph->m_attributesHash = m_scene->attributesPlug()->hash();
			
			// use the precomputed hash in getValue() to save a bit of time:
			
			m_sceneGraph->m_attributes = m_scene->attributesPlug()->getValue( &m_sceneGraph->m_attributesHash );
			const BoolData *visibilityData = m_sceneGraph->m_attributes->member<BoolData>( SceneInterface::visibilityName );
			if( visibilityData && !visibilityData->readable() )
			{
				// terminate recursion for invisible locations
				return NULL;
			}

			if( const BoolData *transformBlurData = m_sceneGraph->m_attributes->member<BoolData>( "gaffer:transformBlur" ) )
			{
				m_sceneGraph->m_transformBlur = transformBlurData->readable();
			}

			if( const IntData *transformBlurSegmentsData = m_sceneGraph->m_attributes->member<IntData>( "gaffer:transformBlurSegments" ) )
			{
				m_sceneGraph->m_transformBlurSegments = transformBlurSegmentsData->readable();
			}

			if( const BoolData *deformationBlurData = m_sceneGraph->m_attributes->member<BoolData>( "gaffer:deformationBlur" ) )
			{
				m_sceneGraph->m_deformationBlur = deformationBlurData->readable();
			}

			if( const IntData *deformationBlurSegmentsData = m_sceneGraph->m_attributes->member<IntData>( "gaffer:deformationBlurSegments" ) )
			{
				m_sceneGraph->m_deformationBlurSegments = deformationBlurSegmentsData->readable();
			}

			// store the hash of the child names so we know when they change:
			m_sceneGraph->m_childNamesHash = m_scene->childNamesPlug()->hash();

			// compute child names:
			IECore::ConstInternedStringVectorDataPtr childNamesData = m_scene->childNamesPlug()->getValue( &m_sceneGraph->m_childNamesHash );

			std::vector<IECore::InternedString> childNames = childNamesData->readable();
			if( childNames.empty() )
			{
				// nothing more to do
				return NULL;
			}
			
			// sort the child names so we can compare child name lists easily in ChildNamesUpdateTask:
			std::sort( childNames.begin(), childNames.end() );

			// add children for this location:
			std::vector<SceneProcedural::SceneGraph *> children;
			for( std::vector<IECore::InternedString>::const_iterator it = childNames.begin(), eIt = childNames.end(); it != eIt; ++it )
			{
				SceneGraph *child = new SceneGraph();
				child->m_name = *it;
				child->m_parent = m_sceneGraph;
				
				child->m_transformBlur = m_sceneGraph->m_transformBlur;
				child->m_transformBlurSegments = m_sceneGraph->m_transformBlurSegments;

				child->m_deformationBlur = m_sceneGraph->m_deformationBlur;
				child->m_deformationBlurSegments = m_sceneGraph->m_deformationBlurSegments;
				
				children.push_back( child );
			}

			// spawn child tasks:
			set_ref_count( 1 + children.size() );
			ScenePlug::ScenePath childPath = m_scenePath;
			childPath.push_back( IECore::InternedString() ); // space for the child name
			for( std::vector<SceneGraph *>::const_iterator it = children.begin(), eIt = children.end(); it != eIt; ++it )
			{
				childPath.back() = (*it)->m_name;
				SceneGraphBuildTask *t = new( allocate_child() ) SceneGraphBuildTask(
					m_scene,
					m_context,
					(*it),
					childPath
				);
				
				spawn( *t );
			}

			wait_for_all();
			
			// add visible children to m_sceneGraph->m_children:
			for( std::vector<SceneGraph *>::const_iterator it = children.begin(), eIt = children.end(); it != eIt; ++it )
			{
				const BoolData *visibilityData = (*it)->m_attributes->member<BoolData>( SceneInterface::visibilityName );
				if( visibilityData && !visibilityData->readable() )
				{
					continue;
				}
				m_sceneGraph->m_children.push_back( *it );
			}
			
			return NULL;
		}

	private :

		const ScenePlug *m_scene;
		const Context *m_context;
		SceneGraph *m_sceneGraph;
		ScenePlug::ScenePath m_scenePath;
};


void SceneProcedural::render( Renderer *renderer ) const
{
	tbb::task_scheduler_init tsi( tbb::task_scheduler_init::deferred );
	initializeTaskScheduler( tsi );

	Context::Scope scopedContext( m_context.get() );

	/// \todo See above.
	try
	{

		// build the scene graph structure in parallel:
		boost::shared_ptr<SceneGraph> sceneGraph( new SceneGraph );
		
		std::cerr << m_attributes.transformBlur << std::endl;
		std::cerr << m_attributes.transformBlurSegments << std::endl;
		std::cerr << m_attributes.deformationBlur << std::endl;
		std::cerr << m_attributes.deformationBlurSegments << std::endl;
		
		std::cerr << m_options.transformBlur << std::endl;
		std::cerr << m_options.deformationBlur << std::endl;
		
		sceneGraph.get()->m_transformBlur = m_attributes.transformBlur;
		sceneGraph.get()->m_transformBlurSegments = m_attributes.transformBlurSegments;
		sceneGraph.get()->m_deformationBlur = m_attributes.deformationBlur;
		sceneGraph.get()->m_deformationBlurSegments = m_attributes.deformationBlurSegments;
		
		std::cerr << "build scene graph" << std::endl;
		SceneGraphBuildTask *task = new( tbb::task::allocate_root() ) SceneGraphBuildTask( m_scenePlug.get(), m_context.get(), sceneGraph.get(), ScenePlug::ScenePath() );
		tbb::task::spawn_root_and_wait( *task );

		// output the scene:
		std::cerr << "output scene" << std::endl;
		outputGeometry( renderer, m_scenePlug.get(), m_context.get(), sceneGraph.get(), m_options.transformBlur, m_options.deformationBlur, m_options.shutter );

		std::cerr << "done" << std::endl;
	}
	catch( const std::exception &e )
	{
		IECore::msg( IECore::Msg::Error, "SceneProcedural::render()", e.what() );
	}
	if( !m_rendered )
	{
		decrementPendingProcedurals();
	}
	m_rendered = true;
}

void SceneProcedural::decrementPendingProcedurals() const
{
	if( --g_pendingSceneProcedurals == 0 )
	{
		try
		{
			tbb::mutex::scoped_lock l( g_allRenderedMutex );
			g_allRenderedSignal();
		}
		catch( const std::exception &e )
		{
			IECore::msg( IECore::Msg::Error, "SceneProcedural::allRenderedSignal() error", e.what() );
		}
	}
}

IECore::MurmurHash SceneProcedural::hash() const
{
	/// \todo Implement me properly.
	return IECore::MurmurHash();
}

void SceneProcedural::updateAttributes( bool full )
{
	Context::Scope scopedContext( m_context.get() );
	
	// \todo: Investigate if it's worth keeping these around and reusing them in SceneProcedural::render().
	
	ConstCompoundObjectPtr attributes;
	if( full )
	{
		attributes = m_scenePlug->fullAttributes( m_scenePath );
	}
	else
	{
		attributes = m_scenePlug->attributesPlug()->getValue();
	}

	if( const BoolData *transformBlurData = attributes->member<BoolData>( "gaffer:transformBlur" ) )
	{
		m_attributes.transformBlur = transformBlurData->readable();
	}

	if( const IntData *transformBlurSegmentsData = attributes->member<IntData>( "gaffer:transformBlurSegments" ) )
	{
		m_attributes.transformBlurSegments = transformBlurSegmentsData->readable();
	}

	if( const BoolData *deformationBlurData = attributes->member<BoolData>( "gaffer:deformationBlur" ) )
	{
		m_attributes.deformationBlur = deformationBlurData->readable();
	}

	if( const IntData *deformationBlurSegmentsData = attributes->member<IntData>( "gaffer:deformationBlurSegments" ) )
	{
		m_attributes.deformationBlurSegments = deformationBlurSegmentsData->readable();
	}
}

void SceneProcedural::motionTimes( unsigned segments, std::set<float> &times ) const
{
	if( !segments )
	{
		times.insert( m_context->getFrame() );
	}
	else
	{
		for( unsigned i = 0; i<segments + 1; i++ )
		{
			times.insert( lerp( m_options.shutter[0], m_options.shutter[1], (float)i / (float)segments ) );
		}
	}
}

SceneProcedural::AllRenderedSignal &SceneProcedural::allRenderedSignal()
{
	return g_allRenderedSignal;
}




//////////////////////////////////////////////////////////////////////////
// ChildNamesUpdateTask implementation
//
// We use this tbb::task to traverse the input scene and check if the child
// names are still valid at each location. If not, we flag the location as
// not present so it doesn't get traversed during an update
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::ChildNamesUpdateTask : public tbb::task
{

	public :

		ChildNamesUpdateTask( const ScenePlug *scene, const Context *context, SceneGraph *sceneGraph, const ScenePlug::ScenePath &scenePath )
			:	m_scene( scene ),
				m_context( context ),
				m_sceneGraph( sceneGraph ),
				m_scenePath( scenePath )
		{
		}

		~ChildNamesUpdateTask()
		{
		}

		virtual task *execute()
		{
			ContextPtr context = new Context( *m_context, Context::Borrowed );
			context->set( ScenePlug::scenePathContextName, m_scenePath );
			Context::Scope scopedContext( context.get() );
			
			IECore::MurmurHash childNamesHash = m_scene->childNamesPlug()->hash();
			
			if( childNamesHash != m_sceneGraph->m_childNamesHash )
			{
				// child names have changed - we need to update m_locationPresent on the children:
				m_sceneGraph->m_childNamesHash = childNamesHash;
				
				// read updated child names:
				IECore::ConstInternedStringVectorDataPtr childNamesData = m_scene->childNamesPlug()->getValue( &m_sceneGraph->m_childNamesHash );
				std::vector<IECore::InternedString> childNames = childNamesData->readable();
				
				// m_sceneGraph->m_children should be sorted by name. Sort this list too so we can
				// compare the two easily:
				std::sort( childNames.begin(), childNames.end() );
				
				std::vector<InternedString>::iterator childNamesBegin = childNames.begin();
				for( std::vector<SceneGraph *>::const_iterator it = m_sceneGraph->m_children.begin(), eIt = m_sceneGraph->m_children.end(); it != eIt; ++it )
				{
					// try and find the current child name in the list of child names:
					std::vector<InternedString>::iterator nameIt = std::find( childNamesBegin, childNames.end(), (*it)->m_name );
					if( nameIt != childNames.end() )
					{
						// ok, it's there - mark this child as still present
						(*it)->m_locationPresent = true;
						
						// As both the name lists are sorted, no further child names will be found beyond nameIt
						// in the list, nor will they be found at nameIt as there shouldn't be any duplicates.
						// This means we can move the start of the child names list one position past nameIt
						// to save a bit of time:
						childNamesBegin = nameIt;
						++childNamesBegin;
					}
					else
					{
						(*it)->m_locationPresent = false;
					}
				}
			}
			
			// count children currently present in the scene:
			size_t numPresentChildren = 0;
			for( std::vector<SceneGraph *>::const_iterator it = m_sceneGraph->m_children.begin(), eIt = m_sceneGraph->m_children.end(); it != eIt; ++it )
			{
				numPresentChildren += (*it)->m_locationPresent;
			}
			
			// spawn child tasks:
			set_ref_count( 1 + numPresentChildren );
			ScenePlug::ScenePath childPath = m_scenePath;
			childPath.push_back( IECore::InternedString() ); // space for the child name
			for( std::vector<SceneGraph *>::const_iterator it = m_sceneGraph->m_children.begin(), eIt = m_sceneGraph->m_children.end(); it != eIt; ++it )
			{
				if( (*it)->m_locationPresent )
				{
					childPath.back() = (*it)->m_name;
					ChildNamesUpdateTask *t = new( allocate_child() ) ChildNamesUpdateTask(
						m_scene,
						m_context,
						(*it),
						childPath
					);

					spawn( *t );
				}
			}
			wait_for_all();

			return NULL;
		}

	private :

		const ScenePlug *m_scene;
		const Context *m_context;
		SceneGraph *m_sceneGraph;
		ScenePlug::ScenePath m_scenePath;
};




//////////////////////////////////////////////////////////////////////////
// SceneGraphIteratorFilter implementation
//
// Does a serial, depth first traversal of a SceneGraph hierarchy based at
// "start", and spits out SceneGraph* tokens:
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::SceneGraphIteratorFilter : public tbb::filter
{
	public:
		SceneGraphIteratorFilter( SceneProcedural::SceneGraph *start ) :
			tbb::filter( tbb::filter::serial_in_order ), m_current( start )
		{
			m_childIndices.push_back( 0 );
		}
		
		virtual void *operator()( void *item )
		{
			++locationCount;
			if( m_childIndices.empty() )
			{
				// we've finished the iteration
				return NULL;
			}
			SceneProcedural::SceneGraph *s = m_current;
			next();
			return s;
		}
	
	private:
		
		void next()
		{
			// go down one level in the hierarchy if we can:
			for( size_t i=0; i < m_current->m_children.size(); ++i )
			{
				// skip out locations that aren't present
				if( m_current->m_children[i]->m_locationPresent )
				{
					m_current = m_current->m_children[i];
					m_childIndices.push_back(i);
					return;
				}
			}
			
			while( m_childIndices.size() )
			{
				
				// increment child index:
				++m_childIndices.back();
				
				// find parent's child count - for the root we define this as 1:
				size_t parentNumChildren = m_current->m_parent ? m_current->m_parent->m_children.size() : 1;
				
				if( m_childIndices.back() == parentNumChildren )
				{
					// we've got to the end of the child list, jump up one level:
					m_childIndices.pop_back();
					m_current = m_current->m_parent;
					continue;
				}
				else if( m_current->m_parent->m_children[ m_childIndices.back() ]->m_locationPresent )
				{
					// move to next child of the parent, if it is still present in the scene:
					m_current = m_current->m_parent->m_children[ m_childIndices.back() ];
					return;
				}
			}
		}
		
		SceneGraph *m_current;
		std::vector<size_t> m_childIndices;
};


//////////////////////////////////////////////////////////////////////////
// SceneGraphEvaluatorFilter implementation
//
// This parallel filter computes the data living at the scene graph
// location it receives. If the "onlyChanged" flag is set to true, it
// only recomputes data when the hashes change, otherwise it computes
// all non null scene data.
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::SceneGraphEvaluatorFilter : public tbb::filter
{
	public:
		SceneGraphEvaluatorFilter(
			const ScenePlug *scene,
			const Context *context, 
			bool transformBlur,
			bool deformationBlur,
			Imath::V2f shutter,
			bool update
		) :
			tbb::filter( tbb::filter::parallel ),
			m_scene( scene ),
			m_context( context ),
			m_transformBlur( transformBlur ),
			m_deformationBlur( deformationBlur ),
			m_shutter( shutter ),
			m_update( update )
		{
			std::cerr << "SceneGraphEvaluatorFilter " << transformBlur << " " << deformationBlur << " " << shutter << std::endl;
		}

		virtual void *operator()( void *item )
		{
			SceneGraph *s = (SceneGraph*)item;
			ScenePlug::ScenePath path;
			s->path( path );

			try
			{
				ContextPtr context = new Context( *m_context, Context::Borrowed );
				context->set( ScenePlug::scenePathContextName, path );
				Context::Scope scopedContext( context.get() );
			
				if( m_update )
				{
					// we're re-traversing this location, so lets only recompute attributes where
					// their hashes change:

					IECore::MurmurHash attributesHash = m_scene->attributesPlug()->hash();
					if( attributesHash != s->m_attributesHash )
					{
						s->m_attributes = m_scene->attributesPlug()->getValue( &attributesHash );
						s->m_attributesHash = attributesHash;
					}
				}
				else
				{
					// First traversal: attributes and attribute hash should have been computed
					// by the SceneGraphBuildTasks, so we only need to compute the object/transform:
					std::set<float> times;

					ContextPtr timeContext = new Context( *context, Context::Borrowed );
					Context::Scope scopedTimeContext( timeContext.get() );

					motionTimes( ( s->m_deformationBlur && m_deformationBlur ) ? s->m_deformationBlurSegments : 0, times );
					for( std::set<float>::const_iterator it = times.begin(), eIt = times.end(); it != eIt; it++ )
					{
						timeContext->setFrame( *it );
						s->m_objectSamples.push_back( std::make_pair( *it, m_scene->objectPlug()->getValue() ) );
					}

					motionTimes( ( s->m_transformBlur && m_transformBlur ) ? s->m_transformBlurSegments : 0, times );
					for( std::set<float>::const_iterator it = times.begin(), eIt = times.end(); it != eIt; it++ )
					{
						timeContext->setFrame( *it );
						s->m_transformSamples.push_back( std::make_pair( *it, m_scene->transformPlug()->getValue() ) );
					}

				}
			}
			catch( const std::exception &e )
			{
				std::string name;
				ScenePlug::pathToString( path, name );
			
				IECore::msg( IECore::Msg::Error, "SceneProcedural::update", name + ": " + e.what() );
			}

			return s;
		}

	private:

		void motionTimes( unsigned segments, std::set<float> &times ) const
		{
			if( !segments )
			{
				times.insert( m_context->getFrame() );
			}
			else
			{
				for( unsigned i = 0; i<segments + 1; i++ )
				{
					times.insert( lerp( m_shutter[0], m_shutter[1], (float)i / (float)segments ) );
				}
			}
		}


		const ScenePlug *m_scene;
		const Context *m_context;
		bool m_transformBlur;
		bool m_deformationBlur;
		Imath::V2f m_shutter;
		const bool m_update;
};

//////////////////////////////////////////////////////////////////////////
// SceneGraphOutputFilter implementation
//
// This serial thread bound filter outputs scene data to a renderer on
// the main thread, then discards that data to save memory. If the
// editMode flag is set to true, the filter outputs edits, otherwise
// it renders the data directly.
//
//////////////////////////////////////////////////////////////////////////

class SceneProcedural::SceneGraphOutputFilter : public tbb::thread_bound_filter
{
	public:
	
		SceneGraphOutputFilter( Renderer *renderer, bool editMode ) :
			tbb::thread_bound_filter( tbb::filter::serial_in_order ), 
			m_renderer( renderer ),
			m_attrBlockCounter( 0 ),
			m_editMode( editMode )
		{
		}
		
		virtual ~SceneGraphOutputFilter()
		{
			// close pending attribute blocks:
			while( m_attrBlockCounter )
			{
				--m_attrBlockCounter;
				m_renderer->attributeEnd();
			}
		}
		
		virtual void *operator()( void *item )
		{
			SceneGraph *s = (SceneGraph*)item;
			ScenePlug::ScenePath path;
			s->path( path );
			
			std::string name;
			ScenePlug::pathToString( path, name );
			
			try
			{
				if( !m_editMode )
				{
					// outputting scene for the first time - do some attribute block tracking:
					if( path.size() )
					{
						for( int i = m_previousPath.size(); i >= (int)path.size(); --i )
						{
							--m_attrBlockCounter;
							m_renderer->attributeEnd();
						}
					}

					m_previousPath = path;

					++m_attrBlockCounter;
					m_renderer->attributeBegin();

					// set the name for this location:
					m_renderer->setAttribute( "name", new StringData( name ) );

				}

				// transform:
				if( !m_editMode )
				{
					if( s->m_transformSamples.size() == 1 )
					{
						m_renderer->concatTransform( s->m_transformSamples[0].second );
					}
					else
					{
						std::set<float> transformTimes;
						for( size_t i=0; i < s->m_transformSamples.size(); ++i )
						{
							transformTimes.insert( s->m_transformSamples[i].first );
						}
						MotionBlock motionBlock( m_renderer, transformTimes );
						for( size_t i=0; i < s->m_transformSamples.size(); ++i )
						{
							m_renderer->concatTransform( s->m_transformSamples[i].second );
						}
					}
				}

				// attributes:
				if( s->m_attributes )
				{
					if( m_editMode )
					{
						CompoundDataMap parameters;
						parameters["exactscopename"] = new StringData( name );
						m_renderer->editBegin( "attribute", parameters );
					}

					for( CompoundObject::ObjectMap::const_iterator it = s->m_attributes->members().begin(), eIt = s->m_attributes->members().end(); it != eIt; it++ )
					{
						if( const StateRenderable *s = runTimeCast<const StateRenderable>( it->second.get() ) )
						{
							s->render( m_renderer );
						}
						else if( const ObjectVector *o = runTimeCast<const ObjectVector>( it->second.get() ) )
						{
							for( ObjectVector::MemberContainer::const_iterator it = o->members().begin(), eIt = o->members().end(); it != eIt; it++ )
							{
								const StateRenderable *s = runTimeCast<const StateRenderable>( it->get() );
								if( s )
								{
									s->render( m_renderer );
								}
							}
						}
						else if( const Data *d = runTimeCast<const Data>( it->second.get() ) )
						{
							m_renderer->setAttribute( it->first, d );
						}
					}
					s->m_attributes = 0;

					if( m_editMode )
					{
						m_renderer->editEnd();
					}

				}

				// object:
				if( s->m_objectSamples.size() && !m_editMode )
				{
					if( s->m_objectSamples[0].second->isInstanceOf( PrimitiveTypeId ) )
					{
						if( s->m_objectSamples.size() == 1 )
						{
							runTimeCast< const Primitive >( s->m_objectSamples[0].second.get() )->render( m_renderer );
						}
						else
						{
							bool hashesDifferent(false);
							std::set<float> deformationTimes;
							for( size_t i=0; i < s->m_objectSamples.size(); ++i )
							{
								if( i > 0 && s->m_objectSamples[i].second->hash() != s->m_objectSamples[0].second->hash() )
								{
									hashesDifferent = true;
								}
								deformationTimes.insert( s->m_objectSamples[i].first );
							}
							if( hashesDifferent )
							{
								MotionBlock motionBlock( m_renderer, deformationTimes );
								for( size_t i=0; i < s->m_objectSamples.size(); ++i )
								{
									runTimeCast< const Primitive >( s->m_objectSamples[i].second.get() )->render( m_renderer );
								}
							}
							else
							{
								runTimeCast< const Primitive >( s->m_objectSamples[0].second.get() )->render( m_renderer );
							}
						}
					}
					else if( const VisibleRenderable *renderable = runTimeCast< const VisibleRenderable >( s->m_objectSamples[0].second.get() ) )
					{
						renderable->render( m_renderer );
					}
					s->m_objectSamples.clear();
				}
			}
			catch( const std::exception &e )
			{
				IECore::msg( IECore::Msg::Error, "SceneProcedural::update", name + ": " + e.what() );
			}

			return NULL;
		}
		
	private:
		
		Renderer *m_renderer;
		ScenePlug::ScenePath m_previousPath;
		int m_attrBlockCounter;
		bool m_editMode;
};


static void runPipeline(tbb::pipeline *p)
{
	// \todo: tune this number to find a balance between memory and speed once
	// we have a load of production data:
	
	p->run( 2 * tbb::task_scheduler_init::default_num_threads() );
}

void SceneProcedural::outputGeometry(
	IECore::Renderer* renderer,
	const ScenePlug* scenePlug,
	const Gaffer::Context* context,
	SceneGraph* sceneGraph, 
	bool transformBlur,
	bool deformationBlur,
	Imath::V2f shutter,
	bool update )
{
	
	SceneGraphIteratorFilter iterator( sceneGraph );

	std::cerr << "trans: " << transformBlur << " def: " << deformationBlur << std::endl;
	SceneGraphEvaluatorFilter evaluator(
		scenePlug,
		context,
		transformBlur,
		deformationBlur,
		shutter,
		update // only recompute locations whose hashes have changed if true:
	);

	SceneGraphOutputFilter output( 
		renderer,
		update // edit mode if true
	);

	tbb::pipeline p;
	p.add_filter( iterator );
	p.add_filter( evaluator );
	p.add_filter( output );

	locationCount = 0;
	 // Another thread initiates execution of the pipeline
	std::thread pipelineThread( runPipeline, &p );

	// Process the SceneGraphOutputFilter with the current thread:
	while( output.process_item() != tbb::thread_bound_filter::end_of_stream )
	{
		continue;
	}
	pipelineThread.join();
	
	std::cerr << locationCount << " locations" << std::endl;
}
