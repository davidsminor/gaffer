##########################################################################
#  
#  Copyright (c) 2013, Image Engine Design Inc. All rights reserved.
#  
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#  
#      * Redistributions of source code must retain the above
#        copyright notice, this list of conditions and the following
#        disclaimer.
#  
#      * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided with
#        the distribution.
#  
#      * Neither the name of John Haddon nor the names of
#        any other contributors to this software may be used to endorse or
#        promote products derived from this software without specific prior
#        written permission.
#  
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#  
##########################################################################

import unittest

import IECore

import Gaffer
import GafferTest

class MetadataTest( GafferTest.TestCase ) :

	class DerivedAddNode( GafferTest.AddNode ) :
		
		def __init__( self, name="DerivedAddNode" ) :

			GafferTest.AddNode.__init__( self, name )
				
	IECore.registerRunTimeTyped( DerivedAddNode )
				
	def testNodeDescription( self ) :
	
		add = GafferTest.AddNode()
	
		self.assertEqual( Gaffer.Metadata.nodeDescription( add ), "" )
	
		Gaffer.Metadata.registerNodeDescription( GafferTest.AddNode, "description" )
		self.assertEqual( Gaffer.Metadata.nodeDescription( add ), "description" )
	
		Gaffer.Metadata.registerNodeDescription( GafferTest.AddNode, lambda node : node.getName() )
		self.assertEqual( Gaffer.Metadata.nodeDescription( add ), "AddNode" )
	
		derivedAdd = self.DerivedAddNode()
		self.assertEqual( Gaffer.Metadata.nodeDescription( derivedAdd ), "DerivedAddNode" )
		self.assertEqual( Gaffer.Metadata.nodeDescription( derivedAdd, inherit=False ), "" )
		
		Gaffer.Metadata.registerNodeDescription( self.DerivedAddNode.staticTypeId(), "a not very helpful description" )
		self.assertEqual( Gaffer.Metadata.nodeDescription( derivedAdd ), "a not very helpful description" )
		self.assertEqual( Gaffer.Metadata.nodeDescription( add ), "AddNode" )
	
	def testExtendedNodeDescription( self ) :
	
		multiply = GafferTest.MultiplyNode()
		
		self.assertEqual( Gaffer.Metadata.nodeDescription( multiply ), "" )
		
		Gaffer.Metadata.registerNodeDescription(
		
			GafferTest.MultiplyNode,
			"description",
			
			"op1",
			"op1 description",
			
			"op2",
			{
				"description" : "op2 description",
				"otherValue" : 100,
			}
			
		)
		
		self.assertEqual( Gaffer.Metadata.nodeDescription( multiply ), "description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( multiply["op1"] ), "op1 description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( multiply["op2"] ), "op2 description" )
		self.assertEqual( Gaffer.Metadata.plugValue( multiply["op2"], "otherValue" ), 100 )
		
	def testPlugDescription( self ) :
	
		add = GafferTest.AddNode()
		
		self.assertEqual( Gaffer.Metadata.plugDescription( add["op1"] ), "" )
		
		Gaffer.Metadata.registerPlugDescription( GafferTest.AddNode.staticTypeId(), "op1", "The first operand" )
		self.assertEqual( Gaffer.Metadata.plugDescription( add["op1"] ), "The first operand" )
		
		Gaffer.Metadata.registerPlugDescription( GafferTest.AddNode.staticTypeId(), "op1", lambda plug : plug.getName() + " description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( add["op1"] ), "op1 description" )
		
		derivedAdd = self.DerivedAddNode()
		self.assertEqual( Gaffer.Metadata.plugDescription( derivedAdd["op1"] ), "op1 description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( derivedAdd["op1"], inherit=False ), "" )
	
		Gaffer.Metadata.registerPlugDescription( self.DerivedAddNode, "op*", "derived class description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( derivedAdd["op1"] ), "derived class description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( derivedAdd["op2"] ), "derived class description" )
		
		self.assertEqual( Gaffer.Metadata.plugDescription( add["op1"] ), "op1 description" )
		self.assertEqual( Gaffer.Metadata.plugDescription( add["op2"] ), "" )
	
	def testArbitraryValues( self ) :

		add = GafferTest.AddNode()

		self.assertEqual( Gaffer.Metadata.nodeValue( add, "aKey" ), None )
		self.assertEqual( Gaffer.Metadata.plugValue( add["op1"], "aKey" ), None )

		Gaffer.Metadata.registerNodeValue( GafferTest.AddNode, "aKey", "something" )
		Gaffer.Metadata.registerPlugValue( GafferTest.AddNode, "op*", "aKey", "somethingElse" )

		self.assertEqual( Gaffer.Metadata.nodeValue( add, "aKey" ), "something" )
		self.assertEqual( Gaffer.Metadata.plugValue( add["op1"], "aKey" ), "somethingElse" )

	def testInheritance( self ) :
	
		Gaffer.Metadata.registerNodeValue( GafferTest.AddNode, "iKey", "Base class value" )
		
		derivedAdd = self.DerivedAddNode()
		self.assertEqual( Gaffer.Metadata.nodeValue( derivedAdd, "iKey" ), "Base class value" )
		self.assertEqual( Gaffer.Metadata.nodeValue( derivedAdd, "iKey", inherit=False ), None )
		
		Gaffer.Metadata.registerNodeValue( self.DerivedAddNode, "iKey", "Derived class value" )
		self.assertEqual( Gaffer.Metadata.nodeValue( derivedAdd, "iKey", inherit=False ), "Derived class value" )
		
		Gaffer.Metadata.registerPlugValue( GafferTest.AddNode, "op1", "iKey", "Base class plug value" )
		self.assertEqual( Gaffer.Metadata.plugValue( derivedAdd["op1"], "iKey" ), "Base class plug value" )
		self.assertEqual( Gaffer.Metadata.plugValue( derivedAdd["op1"], "iKey", inherit=False ), None )

		Gaffer.Metadata.registerPlugValue( self.DerivedAddNode, "op1", "iKey", "Derived class plug value" )
		self.assertEqual( Gaffer.Metadata.plugValue( derivedAdd["op1"], "iKey", inherit=False ), "Derived class plug value" )
		
if __name__ == "__main__":
	unittest.main()

