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
###############################################################################
# XML helper class(es)
#
# Some code is based on the "untangle" package 
# see https://github.com/stchris/untangle and https://untangle.readthedocs.io/en/latest/
# fpr more details

###############################################################################


from .attributes import *
from .dictionaries import *
from .elements import *
from .parser import *


###############################################################################
def serialize(element:XmlElement,/,indent = "", filename=None):
    def serialize_children(strlist,element,indent, base_indent):
        if element.children() is not None:
            if len(element.children()) > 0:
                strlist.append('\n')
                for c in element.children():
                    strlist.append(serialize(c,indent=indent))
                strlist += [base_indent]
        if element.cdata is not None:
                strlist.append(element.cdata)
        return ''.join(strlist)

    strlist = []
    if element.is_root:
        serialize_children(strlist,element,"","")
    else:
        strlist = [indent+'<'+element.original_name()]
        if element.attributes() is not None and len(element.attributes())>0:
            strlist.append(' ')
            attrs = []
            for k,v in element.attributes().original().items():
                attrs+=[k+'="'+v+'"']
            strlist.append(" ".join(attrs))
        strlist.append('>')
    
        serialize_children(strlist,element,indent+"    ",base_indent=indent)

        strlist += ['</',element.original_name(),'>\n']
    result =  ''.join(strlist)
    return result
