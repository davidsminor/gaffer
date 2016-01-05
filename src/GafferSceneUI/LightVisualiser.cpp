//////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2015, John Haddon. All rights reserved.
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

#include "boost/container/flat_map.hpp"
#include "boost/algorithm/string/predicate.hpp"

#include "IECoreGL/CurvesPrimitive.h"
#include "IECoreGL/Group.h"
#include "IECore/Light.h"

#include "GafferSceneUI/LightVisualiser.h"
#include "GafferSceneUI/AttributeVisualiser.h"
#include "GafferSceneUI/StandardLightVisualiser.h"

using namespace std;
using namespace Imath;
using namespace GafferSceneUI;

//////////////////////////////////////////////////////////////////////////
// Internal implementation details
//////////////////////////////////////////////////////////////////////////

namespace
{

typedef boost::container::flat_map<IECore::InternedString, ConstLightVisualiserPtr> LightVisualisers;
LightVisualisers &lightVisualisers()
{
	static LightVisualisers l;
	return l;
}

const LightVisualiser *standardLightVisualiser()
{
	static ConstLightVisualiserPtr l = new StandardLightVisualiser;
	return l.get();
}

/// Class for visualisation of lights. All lights in Gaffer are represented
/// as IECore::Light objects, but we need to visualise them differently
/// depending on their shader name (accessed using `IECore::Light::getName()`). A
/// factory mechanism is provided to map from this type to a specialised
/// LightVisualiser.
class AttributeVisualiserForLights : public AttributeVisualiser
{

	public :

		IE_CORE_DECLAREMEMBERPTR( AttributeVisualiserForLights )

		/// Uses a custom visualisation registered via `registerLightVisualiser()` if one
		/// is available, if not falls back to a basic point light visualisation.
		virtual IECoreGL::ConstRenderablePtr visualise( const IECore::CompoundObject *attributes,
			IECoreGL::ConstStatePtr &state ) const;

	protected :

		static AttributeVisualiser::AttributeVisualiserDescription<AttributeVisualiserForLights> g_visualiserDescription;

};

} // namespace

IECoreGL::ConstRenderablePtr AttributeVisualiserForLights::visualise( const IECore::CompoundObject *attributes,
	IECoreGL::ConstStatePtr &state ) const
{
	if( !attributes )
	{
		return NULL;
	}

	IECoreGL::GroupPtr resultGroup = NULL;
	IECoreGL::StatePtr resultState = NULL;

	/// This seems pretty expensive to do everywhere.
	/// The alternative would be to register attribute visualisers to specific attributes.  But then we wouldn't be able to have a visualiser that is influenced by multiple attributes simultaneously
	for( IECore::CompoundObject::ObjectMap::const_iterator it = attributes->members().begin();
		it != attributes->members().end(); it++ )
	{
		const std::string &key = it->first.string();
		if( !( boost::ends_with( key, ":light" ) || key == "light" ) )
		{
			continue;
		}

		const IECore::ObjectVector *shaderVector = IECore::runTimeCast<const IECore::ObjectVector>( it->second.get() );
		if( !shaderVector || shaderVector->members().size() == 0 ) continue;

		const IECore::Light *lightShader = IECore::runTimeCast<const IECore::Light>(
			 shaderVector->members()[ shaderVector->members().size() - 1 ] ).get();
		if( !lightShader ) continue;


		const LightVisualiser *currentVisualiser = standardLightVisualiser();

		const LightVisualisers &l = lightVisualisers();
		LightVisualisers::const_iterator findVis = l.find( lightShader->getName() );
		if( findVis != l.end() )
		{
			currentVisualiser = findVis->second.get();
		}

		IECoreGL::ConstStatePtr curState = NULL;
		IECoreGL::ConstRenderablePtr curVis = currentVisualiser->visualise( shaderVector, curState );

		if( curVis )
		{
			if( !resultGroup )
			{
				resultGroup = new IECoreGL::Group();
			}
			// resultGroup will be returned as const, so const-casting the children in order to add them
			// is safe
			resultGroup->addChild( const_cast<IECoreGL::Renderable*>( curVis.get() ) );
		}

		if( curState )
		{
			if( !resultState )
			{
				resultState = new IECoreGL::State( false );
			}
			resultState->add( const_cast<IECoreGL::State*>( curState.get() ) );
		}
	}

	state = resultState;
	return resultGroup;
}

AttributeVisualiser::AttributeVisualiserDescription<AttributeVisualiserForLights> AttributeVisualiserForLights::g_visualiserDescription;






//////////////////////////////////////////////////////////////////////////
// LightVisualiser class
//////////////////////////////////////////////////////////////////////////


LightVisualiser::LightVisualiser()
{
}

LightVisualiser::~LightVisualiser()
{
}

void LightVisualiser::registerLightVisualiser( const IECore::InternedString &name, ConstLightVisualiserPtr visualiser )
{
	lightVisualisers()[name] = visualiser;
}