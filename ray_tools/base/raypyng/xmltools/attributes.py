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
class XmlAttribute:
    def __init__(self,xmlvalue=None) -> None:
        if xmlvalue is not None:
            self.__value=xmlvalue

    def set(self,value):
        self.__value = value

    def get(self):
        return self.__value

    def __str__(self):
        return str(self.__value)

    def __repr__(self):
        return f"{self.__class__.__name__}('{self}')"

###############################################################################
class XmlMappedAttribute(XmlAttribute):
    def __init__(self,xmlvalue=None,map=None) -> None:
        if map is None:
            raise ValueError(f"map parameter is required and can not be 'None'")
        self.__map = map
        super().__init__(xmlvalue)

        if xmlvalue in self.__map:
            self.set(self.__map[xmlvalue] )
        else:
            raise ValueError(f"Invalid value for the XmlAttribute: {xmlvalue}")


    def __str__(self):
        for k,v in self.__map.items():
            if v==self.get():
                return k
        raise ValueError(f"Can not map to bool: {self.get()}")

    def __repr__(self):
        return f"{self.__class__.__name__}('{self}',map={self.__map})"


###############################################################################
class XmlBoolAttribute(XmlMappedAttribute):
    def __init__(self,xmlvalue=None,true='True',false='False') -> None:
        m = {true:True,false:False}
        super().__init__(xmlvalue,m)

