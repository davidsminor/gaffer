##########################################################################
#  
#  Copyright (c) 2012, John Haddon. All rights reserved.
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
import inspect
import types

import IECore

import Gaffer

## A useful base class for creating test cases for nodes.
class TestCase( unittest.TestCase ) :

	## Attempts to ensure that the hashes for a node
	# are reasonable by jiggling around input values
	# and checking that the hash changes when it should.
	def assertHashesValid( self, node, inputsToIgnore=[] ) :
	
		# find all input ValuePlugs
		inputPlugs = []
		def __walkInputs( parent ) :
			for child in parent.children() :
				if isinstance( child, Gaffer.CompoundPlug ) :
					__walkInputs( child )
				elif isinstance( child, Gaffer.ValuePlug ) :
					ignore = False
					for toIgnore in inputsToIgnore :
						if child.isSame( toIgnore ) :
							ignore = True
							break
					if not ignore :
						inputPlugs.append( child )
		__walkInputs( node )
		
		self.failUnless( len( inputPlugs ) > 0 )
		
		numTests = 0
		for inputPlug in inputPlugs :
			for outputPlug in node.affects( inputPlug ) :
				
				hash = outputPlug.hash()
				
				value = inputPlug.getValue()
				if isinstance( value, float ) :
					increment = 0.1
				elif isinstance( value, int ) :
					increment = 1
				elif isinstance( value, basestring ) :
					increment = "a"
				else :
					# don't know how to deal with this
					# value type.
					continue
					
				inputPlug.setValue( value + increment )
				if inputPlug.getValue() == value :
					inputPlug.setValue( value - increment )
				if inputPlug.getValue() == value :
					continue
			
				self.assertNotEqual( outputPlug.hash(), hash )
				
				numTests += 1
				
		self.failUnless( numTests > 0 )
		
	def assertTypeNamesArePrefixed( self, module, namesToIgnore = () ) :
	
		for name in dir( module ) :
		
			cls = getattr( module, name )
			if not inspect.isclass( cls ) :
				continue
				
			if issubclass( cls, IECore.RunTimeTyped ) :
				if cls.staticTypeName() in namesToIgnore :
					continue
				self.assertEqual( cls.staticTypeName(), module.__name__ + "::" + cls.__name__ )
		
	def assertDefaultNamesAreCorrect( self, module ) :
	
		for name in dir( module ) :
		
			cls = getattr( module, name )
			if not inspect.isclass( cls ) or not issubclass( cls, Gaffer.GraphComponent ) :
				continue
			
			try :
				instance = cls()
			except :
				continue

			self.assertEqual( instance.getName(), cls.staticTypeName().rpartition( ":" )[2] )
