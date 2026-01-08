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
from ..collections import MappedList, MappedDict
import keyword


###############################################################################
def sanitizeName(name:str)->str:
    """convert name into python attribute safe name

    Args:
        name (str): _description_

    Returns:
        str: _description_
    """

    if name is None:
        return None

    # repalce special characters with _
    name = name.replace("-", "_")
    name = name.replace(".", "_")
    name = name.replace(":", "_")

    # delete spaces
    name = name.replace(" ", "")

    # adding trailing _ for keywords
    if keyword.iskeyword(name):
        name += "_"
    return name

###############################################################################
class SafeValueDict(MappedDict):
    def __init__(self, dict=None, **kwargs):
        super().__init__(sanitizeName, dict, **kwargs)

###############################################################################
class SafeValueList(MappedList):
    def __init__(self, initlist=None):
        super().__init__(sanitizeName, initlist)


