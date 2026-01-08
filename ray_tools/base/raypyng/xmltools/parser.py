#The MIT License (MIT)
#
#Copyright (c) <2022> <Simone Vadilonga, Ruslan Ovsyannikov>
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#THE SOFTWARE.
from multiprocessing import parent_process
from xml.sax import make_parser, handler
from .elements import *
from .dictionaries import *


###############################################################################
global_known_classes = {} # part of the development code
class Handler(handler.ContentHandler):
    
    #####################################
    def __init__(self,/,known_classes=None):
        self.root = XmlElement(None, None)
        self.root.is_root = True
        self.elements = []
        if known_classes is None:
            self._known_classes = global_known_classes
        else:
            self._known_classes = known_classes


    #####################################
    def startElement(self, name, attributes):
        """called on the start of an element in non-namespace mode.

        Args:
            name (_type_): _description_
            attributes (_type_): _description_
        """
        #print("DEBUG::startElement::name=",name)

        # store attributes in a dictionary
        # local copy is nessesary, input object can change and we should not 
        # save reference to it

        # Shall it be moved into the 
        attrs = SafeValueDict()
        for k, v in attributes.items():
            attrs[k] = v#self.protectName(v)
        
        # create a new element
        if len(self.elements) > 0:
            parent = self.elements[-1]
        else:
            parent = self.root

        if name in self._known_classes.keys():
            element = self._known_classes[name](name, attrs, parent=parent)
        else:
            element = XmlElement(name, attrs,paren=parent)
        # and add it to the known element list
        parent.add_child(element)
        self.elements.append(element)

    #####################################
    def endElement(self, name):
        """called the end of an element in non-namespace mode.

        Args:
            name (_type_): _description_
        """
        self.elements.pop()

    #####################################
    def characters(self, cdata):
        """adds character data to the current element

        Args:
            cdata (_type_): _description_
        """
        self.elements[-1].add_cdata(cdata.strip())

###############################################################################
def parse(filename:str, /, known_classes = None, **parser_features)->XmlElement:
    if filename is None:
        raise ValueError("parse() takes a filename")
    parser = make_parser()
    for feature, value in parser_features.items():
        parser.setFeature(getattr(handler, feature), value)
    sax_handler = Handler(known_classes=known_classes)
    parser.setContentHandler(sax_handler)
    parser.parse(filename)
    return sax_handler.root

