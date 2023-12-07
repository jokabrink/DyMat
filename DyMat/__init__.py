# Copyright (c) 2011, Joerg Raedler (Berlin, Germany)
#               2023, Jonas Kock am Brink
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer. Redistributions in binary form must
# reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the
# distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__version__ = "0.7"
__author__ = "Joerg Raedler (joerg@j-raedler.de)"
__license__ = "BSD License (http://www.opensource.org/licenses/bsd-license.php)"

import math
import os
import sys
from math import copysign
from typing import Union

import numpy
from scipy.io import loadmat


def array2strings(arr: numpy.ndarray) -> list[str]:
    """Convert an array of row char vectors to a list of strings.
    >>> arr = numpy.array([["T", "e", "s", "t"], ["h", "i", "", ""]])
    >>> array2strings(arr)
    ['Test', 'hi']

    >>> arr = numpy.array([["T", "h"], ["e", "i"], ["s", ""], ["t", ""]])
    >>> array2strings(arr.T)
    ['Test', 'hi']

    This is usually for `name` and `description`
    """
    width = arr.shape[-1]
    view = numpy.ascontiguousarray(arr).view(dtype=f"U{width}")
    res = [str(numpy.char.rstrip(x[0])) for x in view]
    return res


class DyMatFileError(Exception):
    pass


class DyMatFile:
    """A result file written by Dymola or OpenModelica"""

    def __init__(
        self,
        mat: dict[str, numpy.ndarray],
        variables: dict[str, tuple[str, int, int, float]],
        blocks: list[int],
        abscissa: tuple[str, str],
    ):
        self.mat = mat
        self._vars = variables  # name: (description, blocknum, column, sign)
        self._blocks = blocks  # block_num: np.ndarray with shape (num_points, num_variables)
        self._absc = abscissa

    def blocks(self):
        """Returns the numbers of all data blocks.

        :Arguments:
            - None
        :Returns:
            - sequence of integers
        """
        return self._blocks

    def names(self, block=None):
        """Returns the names of all variables. If block is given, only variables of this
        block are listed.

        :Arguments:
            - optional block: integer
        :Returns:
            - sequence of strings
        """
        if block is None:
            return self._vars.keys()
        else:
            return [k for (k, v) in self._vars.items() if v[1] == block]

    def data(self, varName):
        """Return the values of the variable.

        :Arguments:
            - varName: string
        :Returns:
            - numpy.ndarray with the values"""
        tmp, d, c, s = self._vars[varName]
        di = "data_%d" % (d)
        dd = self.mat[di][c]
        if s < 0:
            dd = dd * -1
        return dd

    # add a dictionary-like interface
    __getitem__ = data

    def block(self, varName):
        """Returns the block number of the variable.

        :Arguments:
            - varName: string
        :Returns:
            - integer
        """
        return self._vars[varName][1]

    def description(self, varName):
        """Returns the description string of the variable.

        :Arguments:
            - varName: string
        :Returns:
            - string
        """
        return self._vars[varName][0]

    def sharedData(self, varName):
        """Return variables which share data with this variable, possibly with a different
        sign.

        :Arguments:
            - varName: string
        :Returns:
            - sequence of tuples, each containing a string (name) and a number (sign)
        ."""
        tmp, d, c, s = self._vars[varName]
        return [
            (n, v[3] * s)
            for (n, v) in self._vars.items()
            if n != varName and v[1] == d and v[2] == c
        ]

    def size(self, blockOrName):
        """Return the number of rows (time steps) of a variable or a block.

        :Arguments:
            - integer (block) or string (variable name): blockOrName
        :Returns:
            - integer
        """
        try:
            b = int(blockOrName)
        except:
            b = self._vars[blockOrName][1]
        di = "data_%d" % (b)
        return self.mat[di].shape[1]

    def abscissa(self, blockOrName, valuesOnly=False):
        """Return the values, name and description of the abscissa that belongs to a
        variable or block. If valuesOnly is true, only the values are returned.

        :Arguments:
            - integer (block) or string (variable name): blockOrName
            - optional bool: valuesOnly
        :Returns:
            - numpy.ndarray: values or
            - tuple of numpy.ndarray (values), string (name), string (description)
        """
        try:
            b = int(blockOrName)
        except:
            b = self._vars[blockOrName][1]
        di = "data_%d" % (b)
        if valuesOnly:
            return self.mat[di][0]
        else:
            return (
                self.mat[di][0],
                self._absc[0],
                self._absc[1],
            )

    def sortByBlocks(self, varList):
        """Sort a list of variables by the block number, return a dictionary whose keys
        are the block numbers and the values are lists of names. All variables in one
        list will have the same number of values.

        :Arguments:
            - list of strings: varList
        :Returns:
            - dictionary with integer keys and string lists as values
        """
        vl = [(v, self._vars[v][1]) for v in varList]
        vDict = {}
        for bl in self._blocks:
            l = [v for v, b in vl if b == bl]
            if l:
                vDict[bl] = l
        return vDict

    def nameTree(self):
        """Return a tree of all variable names with respect to the path names. Path
        elements are separated by dots. The tree will represent the structure of the
        Modelica models. The tree is returned as a dictionary of dictionaries. The keys
        are the path elements, values are sub-dictionaries or variable names.

        :Arguments:
            - None
        :Returns:
            - dictionary
        """
        root = {}
        for v in self._vars.keys():
            branch = root
            elem = v.split(".")
            for e in elem[:-1]:
                if not e in branch:
                    branch[e] = {}
                branch = branch[e]
            branch[elem[-1]] = v
        return root

    def getVarArray(self, varNames, withAbscissa=True):
        """Return the values of all variables in varNames combined as a 2d-array. If
        withAbscissa is True, include abscissa's values first.
        **All variables must share the same block!**

        :Arguments:
            - sequence of strings: varNames
            - optional bool: withAbscissa
        :Returns:
            - numpy.ndarray
        """
        # FIXME: check blocks!
        v = [numpy.array(self.data(n), ndmin=2) for n in varNames]
        if withAbscissa:
            v.insert(
                0,
                numpy.array(
                    self.abscissa(varNames[0], True),
                    ndmin=2,
                ),
            )
        return numpy.concatenate(v, 0)

    def writeVar(self, varName):
        """Write the values of the abscissa and the variabale to stdout. The text format
        is compatible with gnuplot. For more options use DyMat.Export instead.

        :Arguments:
            - string: varName
        :Returns:
            - None
        """
        d = self.data(varName)
        a, aname, tmp = self.abscissa(varName)
        print("# %s | %s" % (aname, varName))
        for i in range(d.shape[0]):
            print("%f %g" % (a[i], d[i]))


def _load_v1_1(
    mat: dict[str, numpy.ndarray],
    transpose: bool = False,
) -> DyMatFile:
    """There are two types: 'BinTrans' and 'BinNormal'
    - BinTrans: Shape is (variables, time). Usually files from OpenModelica or Dymola auto save, all
      methods rely on this structure since this was the only understood by earlier versions
    - BinNormal: Shape is (time, variables). Usually saved files from Dymola, it is converted to
      BinTrans representation

    Source:
    <https://www.claytex.com/tech-blog/trajectory-file-what-is-it-dissecting-a-dymola-result-file/>
    """
    variables: dict[str, tuple[str, int, int, float]] = {}
    blocks: list[int] = []
    if transpose is True:
        names = array2strings(mat["name"].T)
        descriptions = array2strings(mat["description"].T)
        dataInfo = mat["dataInfo"].T
    else:
        names = array2strings(mat["name"])
        descriptions = array2strings(mat["description"])
        dataInfo = mat["dataInfo"]

    assert (
        len(dataInfo) == len(names) == len(descriptions)
    ), f"lengths of `dataInfo`, `names` and `descriptions` do not match. They are {len(dataInfo)}, {len(names)} and {len(descriptions)}"

    for info, name, description in zip(dataInfo, names, descriptions):
        # dataInfo[k] yields a 4-tuple with:
        # k=0: blocknum
        # k=1: value
        # k=2: Linear interpolation of column data
        # k=3: == -1: name not defined outside time range
        #      ==  0: keep first/last value outside of time range
        #      ==  1: linear interpolation through first/last two points outside of time range
        blocknum, value, *_ = info
        column = abs(value) - 1
        sign = copysign(1.0, value)
        if column:
            variables[name] = (
                description,
                blocknum,
                column,
                sign,
            )
            if not blocknum in blocks:
                blocks.append(blocknum)
                if transpose is False:
                    b = f"data_{blocknum}"
                    mat[b] = mat[b].transpose()
        else:
            abscissa = (name, description)
    return DyMatFile(
        mat=mat,
        variables=variables,
        blocks=blocks,
        abscissa=abscissa,
    )


def _load_v1_0(
    mat: dict[str, numpy.ndarray],
) -> DyMatFile:
    # files generated with dymola, save as..., only plotted ...
    # fake the structure of a 1.1 transposed file
    # keys of mat file: "Aclass", "names", "data"
    names = array2strings(mat["names"])
    variables: dict[str, tuple[str, int, int, float]] = {}
    mat["data_0"] = mat["data"].transpose()
    del mat["data"]
    # TODO: Change for loop with enumerate(names)?
    for i in range(1, len(names)):
        variables[names[i]] = ("", 0, i, 1)
    return DyMatFile(
        mat=mat,
        variables=variables,
        blocks=[0],
        abscissa=(names[0], ""),
    )


def load(fileName: str) -> DyMatFile:
    mat = loadmat(fileName, matlab_compatible=True, chars_as_strings=False)

    if "Aclass" not in mat:
        raise DyMatFileError("file does not have 'Aclass' variable")

    fileInfo = array2strings(mat["Aclass"])

    if fileInfo[1] == "1.1":
        if fileInfo[3] == "binNormal":
            return _load_v1_1(mat)
        elif fileInfo[3] == "binTrans":
            # binTrans means the data is in (measurements, time) order, transpose it.
            return _load_v1_1(mat, transpose=True)
        else:
            raise DyMatFileError(f"invalid file structure representation: '{fileInfo[3]}'")
    elif fileInfo[1] == "1.0":
        if fileInfo[3] != "binNormal":
            raise DyMatFileError(f"File version 1.0 not `binNormal`: '{fileInfo[3]}'")
        return _load_v1_0(mat)
    else:
        raise DyMatFileError(f"invalid file version: '{fileInfo[1]}'")
