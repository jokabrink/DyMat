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

from __future__ import annotations

import re
from math import copysign
from numbers import Integral
from typing import Any, Optional, Union

import numpy
from scipy.io import loadmat

__version__ = "0.8"
__author__ = "Jonas Kock am Brink (jokabrink@posteo.de)"
__license__ = "BSD License (http://www.opensource.org/licenses/bsd-license.php)"


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
    res = list(map(lambda s: numpy.char.rstrip(bytes(s[0], "latin-1")).item().decode(), view))
    return res


def _collect(
    value: Union[str, list[str]],
    *values: str,
) -> list[str]:
    """Collect value and values in a flat array
    >>> _collect("a", "b", "c", "d")
    ['a', 'b', 'c', 'd']

    >>> _collect(["a", "b"], "c", "d")
    ['a', 'b', 'c', 'd']
    """
    if type(value) is str:
        return [value, *values]
    else:
        return [*value, *values]


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
        self._blocks = blocks  # [block_num]
        self._absc = abscissa  # [(name, description)]

    def blocks(self) -> list[int]:
        """Returns the numbers of all data blocks."""
        return self._blocks

    def names(self, pattern: Optional[str] = None) -> list[str]:
        """All available variable names or a subset given the regex.
        I.e.: list(filter(lambda n: n=="some.comp.value")) == names("^some.comp.value$")
        """
        if pattern is None:
            return list(self._vars.keys())
        else:
            return self.regex(pattern)

    def descriptions(
        self,
        name: Union[str, list[str]],
        *names: str,
    ) -> list[str]:
        """Given multiple names, return their description in a list."""
        all_names = _collect(name, *names)
        return [self._vars[var_name][0] for var_name in all_names]

    def data(self, name: str) -> numpy.ndarray:
        """Return the values of the variable."""
        _, blocknum, column, sign = self._vars[name]
        dd: numpy.ndarray = self.mat[f"data_{blocknum}"][column]
        if sign < 0:
            dd = dd * -1
        return dd

    def regex(
        self,
        pattern: Union[str, list[str]],
        *patterns: str,
        prefixes: Optional[list[str]] = None,
    ) -> list[str]:
        """
        prefixes: List of
        E.g.: regex("epp\\d.P$", prefixes=["bus1", "bus2"]) \
              == ["bus1.epp1.P", "bus1.epp2.P", "bus2.epp1.P", "bus2.epp2.P"]
        """
        all_names = _collect(pattern, *patterns)
        if prefixes is None:
            regex_func = re.compile("|".join(all_names))
        else:
            s = "^(%s)(%s)" % ("|".join(prefixes), "|".join(all_names))
            regex_func = re.compile(s)

        filtered_names = filter(lambda e: regex_func.match(e), self._vars.keys())
        return list(filtered_names)

    # add a dictionary-like interface
    __getitem__ = data

    def block(self, name: str) -> int:
        """Returns the block number of the variable."""
        return self._vars[name][1]

    def description(self, name: str) -> str:
        """Return the description of a single variable name."""
        return self._vars[name][0]

    def sharedData(self, name: str) -> list[tuple[str, float]]:
        """Return variables which share data with this variable, possibly with a different
        sign."""
        _, blocknum, column, sign = self._vars[name]
        res = []
        for name, v in self._vars.items():
            if name != name and v[1] == blocknum and v[2] == column:
                res.append((name, v[3] * sign))
        return res

    def size(self, blockOrName: Union[int, str]) -> int:
        """Return the number of rows (time steps) of a variable or a block."""
        if isinstance(blockOrName, str):
            b = self._vars[blockOrName][1]
        else:
            b = blockOrName

        return self.mat[f"data_{b}"].shape[1]

    def abscissa(
        self,
        blockOrName: Union[int, str],
        valuesOnly: bool = False,
    ):
        """Return the values, name and description of the abscissa that belongs to a
        variable or block. If valuesOnly is true, only the values are returned.

        :Returns:
            - numpy.ndarray: values or
            - tuple of numpy.ndarray (values), string (name), string (description)
        """
        if isinstance(blockOrName, str):
            b = self._vars[blockOrName][1]
        else:
            b = blockOrName

        di = f"data_{b}"
        if valuesOnly:
            return self.mat[di][0]
        else:
            return (
                self.mat[di][0],
                self._absc[0],
                self._absc[1],
            )

    def sortByBlocks(self, varList: list[str]) -> dict[int, list[str]]:
        """Sort a list of variables by the block number, return a dictionary whose keys
        are the block numbers and the values are lists of names. All variables in one
        list will have the same number of values.
        """
        vl = [(v, self._vars[v][1]) for v in varList]
        vDict = {}
        for bl in self._blocks:
            l = [v for v, b in vl if b == bl]
            if l:
                vDict[bl] = l
        return vDict

    def nameTree(self) -> dict[str, Any]:
        """Return a tree of all variable names with respect to the path names. Path
        elements are separated by dots. The tree will represent the structure of the
        Modelica models. The tree is returned as a dictionary of dictionaries. The keys
        are the path elements, values are sub-dictionaries or variable names.
        """
        root: dict[str, Any] = {}
        for v in self._vars.keys():
            branch = root
            elem = v.split(".")
            for e in elem[:-1]:
                if not e in branch:
                    branch[e] = {}
                branch = branch[e]
            branch[elem[-1]] = v
        return root

    def getVarArray(
        self,
        names: list[str],
        withAbscissa: bool = True,
    ) -> numpy.ndarray:
        """Return the values of all variables in names combined as a 2d-array. If
        withAbscissa is True, include abscissa's values first.
        **All variables must share the same block!**
        """
        # FIXME: check blocks!
        v = [numpy.array(self.data(n), ndmin=2) for n in names]
        if withAbscissa:
            v.insert(
                0,
                numpy.array(
                    self.abscissa(names[0], True),
                    ndmin=2,
                ),
            )
        return numpy.concatenate(v, 0)

    def writeVar(self, name: str) -> None:
        """Write the values of the abscissa and the variabale to stdout. The text format
        is compatible with gnuplot. For more options use DyMat.Export instead.
        """
        d = self.data(name)
        a, aname, _ = self.abscissa(name)
        print("# %s | %s" % (aname, name))
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


def _load_v1_0(mat: dict[str, numpy.ndarray]) -> DyMatFile:
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


def _load(mat: dict[str, numpy.ndarray]) -> DyMatFile:
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


def load(fileName: str) -> DyMatFile:
    """Load a trajectory result file from Dymola or OpenModelica"""
    mat = loadmat(fileName, matlab_compatible=True)
    return _load(mat)
