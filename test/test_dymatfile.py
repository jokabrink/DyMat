import DyMat
import numpy as np
import pytest

_FILES = (
    "data_v1_1_binTrans.mat",
    # "data_v1_1_binNormal.mat",
    # "data_v1_0_binNormal.mat",
)


@pytest.fixture(scope="module", params=_FILES)
def dmat(request):
    yield DyMat.load(request.param)


def create_aclass(version: int = 1, transpose: bool = True):
    traj = list("Atrajectory")
    length = len(traj)

    ver = list(f"1.{version}") + [""]
    ver += (length - len(ver)) * [" "]

    empty = length * [" "]
    empty[1] = ""

    if transpose:
        layout = list("binTrans") + [""]
        layout += (length - len(layout)) * [" "]
    else:
        layout = list("binNormal") + [""]
        layout += (length - len(layout)) * [" "]

    return np.array([traj, ver, empty, layout], dtype="<U1")


def strings2array(strings):
    length = max(len(n) for n in strings)
    lst = [list(n) + (length - len(n)) * [" "] for n in strings]
    return np.array(lst, order="C", copy=True)


@pytest.fixture
def create_mat():
    def _create_mat(version=1, transpose: bool = True):
        Aclass = create_aclass(version=version, transpose=transpose)

        names = ["Time", "x", "der(x)", "v", "alpha"]
        descriptions = ["Time in [s]", "value", "der(value)", "", "time constant"]
        dataInfos = [[0, 1, 0, -1], [2, 2, 0, -1], [2, 3, 0, -1], [2, 3, 0, -1], [1, 2, 0, -1]]

        if transpose is True:
            name = np.copy(strings2array(names).T)
            description = np.copy(strings2array(descriptions).T)
            dataInfo = np.array(dataInfos, dtype=np.int32).T
        else:
            name = np.copy(strings2array(names))
            description = np.copy(strings2array(descriptions))
            dataInfo = np.array(dataInfos, dtype=np.int32)

        data_1 = np.array([[0, 10], [1, 1]], dtype=np.float32)
        data_2 = np.array([], dtype=np.float32)

        mat = {
            "Aclass": Aclass,
            "name": name,
            "description": description,
            "dataInfo": dataInfo,
            "data_1": data_1,
            "data_2": data_2,
        }
        return mat

    return _create_mat


def test_load():
    dmat = DyMat.load("data_v1_1_binTrans.mat")
    # dmat = DyMat.load("data_v1_1_binNormal.mat")
    dmat = DyMat.load("data_v1_0_binNormal.mat")


def test_array2strings():
    arr = np.array([["T", "e", "s", "t"], ["h", "i", "", " "]])
    strings = DyMat.array2strings(arr)
    assert strings == ["Test", "hi"]

    arr = np.array([["T", "h"], ["e", "i"], ["s", " "], ["t", " "]])
    strings = DyMat.array2strings(arr.T)
    assert strings == ["Test", "hi"]

    arr = np.array([["Ã", "¤", " "], ["â", "\x9a", "¡"]])
    strings = DyMat.array2strings(arr)
    assert strings == ["ä", "⚡"]


def test__collect():
    lst = DyMat._collect("a", "b", "c", "d")
    assert lst == ["a", "b", "c", "d"]

    lst = DyMat._collect(["a", "b"], "c", "d")
    assert lst == ["a", "b", "c", "d"]


def test__regex():
    names = ["a1.b", "a2.b", "a3.b", "b.b", "c.b"]
    n = DyMat._regex(names, "a")
    assert n == ["a1.b", "a2.b", "a3.b"]


def test_blocks(dmat):
    # assert dmat.blocks() == [2, 1]
    assert set(dmat.blocks()) == {2, 1}


def test_abscissa(dmat):
    # TODO: compare abscissa(1) with abscissa(alpha)
    values_1 = dmat.abscissa(1, valuesOnly=True)
    values, name, description = dmat.abscissa(1)
    assert np.all(np.isclose(values - values_1, 0))
    assert np.all(np.isclose(values - np.array([0.0, 10.0]), 0))
    assert name == "Time"
    assert description == "Time in [s]"


def test_descriptions(dmat):
    assert dmat.descriptions("alpha") == ["time constant"]
    assert dmat.descriptions(["v"]) == ["velocity ⚡"]
    assert dmat.descriptions(["alpha"], "x") == ["time constant", "välue"]


def test_time(dmat):
    time_1 = [0, 10]
    time_2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]
    time = dmat.time("alpha")
    assert np.all(np.isclose(time, time_1))
    time = dmat.time("x")
    assert np.all(np.isclose(time, time_2))
    time = dmat.time()
    assert np.all(np.isclose(time, time_2))


def test_time_data(dmat):
    time_1 = np.array([0, 10])
    data_1 = np.array([[1, 1]]).T
    time, data = dmat.time_data("alpha")
    assert time.shape == time_1.shape
    assert np.all(np.isclose(time, time_1))
    assert data.shape == data_1.shape
    assert np.all(np.isclose(data, data_1))

    time_2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10], dtype=float)
    time, data = dmat.time_data("x")
    data_2 = np.exp(-1 * time).reshape((-1, 1))
    assert time.shape == time_2.shape
    assert np.all(np.isclose(time, time_2))
    assert data.shape == data_2.shape
    assert np.all(np.isclose(data, data_2, atol=1e-5, rtol=1e-1))

    data_3 = np.stack([np.exp(-1 * time), np.ones(len(time))], axis=1)
    time, data = dmat.time_data("x", "alpha")
    assert data.shape == data_3.shape
    assert np.all(np.isclose(data, data_3, atol=1e-5, rtol=1e-1))


def test_data(dmat):
    assert np.all(np.isclose(dmat.data("alpha"), 1.0))

    time = dmat.abscissa(2, valuesOnly=True)
    x = np.exp(-1 * time).reshape((-1, 1))
    data = dmat.data("x").reshape((-1, 1))
    assert x.shape == data.shape
    assert np.all(np.isclose(data, x, atol=1e-5, rtol=1e-1))


def test_block(dmat):
    assert dmat.block("alpha") == 1
    assert dmat.block("x") == 2


def test_description(dmat):
    assert dmat.description("alpha") == "time constant"


def test_sharedData(dmat):
    assert dmat.sharedData("alpha") == []
    assert dmat.sharedData("v") == [("der(x)", 1.0)]


def test_size(dmat):
    with pytest.raises(KeyError):
        dmat.size(0)

    assert dmat.size(1) == len(dmat.abscissa(1, valuesOnly=True))
    assert dmat.size(2) == len(dmat.abscissa(2, valuesOnly=True))
    assert dmat.size("x") == len(dmat.abscissa("x", valuesOnly=True))


def test_names(dmat):
    names = {"x", "der(x)", "alpha"}
    assert names.issubset(dmat.names())

    names = {"alpha"}
    assert names.issubset(dmat.names("alpha"))

    assert dmat.names("x") == ["x"]
