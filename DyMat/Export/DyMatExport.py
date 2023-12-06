#!/usr/bin/env python

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

import argparse
import sys

import DyMat


def run(args):
    import DyMat

    dm = DyMat.DyMatFile(args.matfile)

    if args.info:
        blocks = dm.blocks()
        blocks.sort()
        for b in blocks:
            print("Block %02d:" % b)
            s = dm.mat["data_%d" % (b)].shape
            v = len(dm.names(b))
            print("  %d variables point to %d columns with %d timesteps" % (v, s[0] - 1, s[1]))

    elif args.list:
        for n in dm.names():
            print(n)

    elif args.descriptions:
        nn = dm.names()
        nlen = max((len(n) for n in nn))
        for n in dm.names():
            print(
                "%s | %02d | %s"
                % (
                    n.ljust(nlen),
                    dm.block(n),
                    dm.description(n),
                )
            )

    elif args.tree:
        t = dm.nameTree()

        def printBranch(branch, level):
            if level > 0:
                tmp = "  |" * level + "--"
            else:
                tmp = "--"
            for elem in branch:
                sub = branch[elem]
                if isinstance(sub, dict):
                    print(tmp + elem)
                    printBranch(sub, level + 1)
                else:
                    print("%s%s (%s)" % (tmp, elem, sub))

        printBranch(t, 0)

    elif args.shared_data:
        v = args.shared_data
        sd = dm.sharedData(v)
        if sd:
            print(v)
            for n, s in sd:
                print("    = % 2d * %s" % (s, n))

    # FIXME: this should work without providing a filename
    elif args.list_formats:
        import DyMat.Export

        for n in DyMat.Export.formats:
            print("%s : %s" % (n, DyMat.Export.formats[n]))

    else:  # args.export or args.export_file
        if args.export:
            varList = [v.strip() for v in args.export.split(",")]
        else:
            varList = [l.split("|")[0].strip() for l in open(args.export_file, "r") if l]

        options = {}
        if args.options:
            tmp = [v.strip().split("=") for v in args.options.split(",")]
            for x in tmp:
                options[x[0]] = x[1]
        import DyMat.Export

        fmt = "CSV" if args.format is None else args.format

        DyMat.Export.export(fmt, dm, varList, args.outfile, options)


def main(argv=None):
    parser = argparse.ArgumentParser()

    grp = parser.add_mutually_exclusive_group(required=True)

    grp.add_argument(
        "-i",
        "--info",
        action="store_true",
        help="show some information on the file",
    )
    grp.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list variables",
    )
    grp.add_argument(
        "-d",
        "--descriptions",
        action="store_true",
        help="list variables with descriptions",
    )
    grp.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="list variables as name tree",
    )
    grp.add_argument(
        "-s",
        "--shared-data",
        metavar="VAR",
        help="list connections of variable",
    )
    grp.add_argument(
        "-m",
        "--list-formats",
        action="store_true",
        help="list supported export formats",
    )
    grp.add_argument(
        "-e",
        "--export",
        metavar="VARLIST",
        help="export these variables",
    )
    grp.add_argument(
        "-x",
        "--export-file",
        metavar="FILE",
        help="export variables listed in this file",
    )

    parser.add_argument(
        "-o",
        "--outfile",
        help="write exported data to this file",
    )
    parser.add_argument(
        "-f",
        "--format",
        help="export data in this format",
    )
    parser.add_argument(
        "-p",
        "--options",
        help="export options specific to export format",
    )

    parser.add_argument("matfile", help="MAT-file")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
