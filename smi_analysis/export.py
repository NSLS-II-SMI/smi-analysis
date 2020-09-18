import h5py
import os
import inspect
import datetime

def store_saxs_2d(path=None, filename='test', img=None, qpar=None, qver=None, wav=0.077, sdd=273.9, beam_shape='200x20'):
    '''
    Save SMI data, already converted in q-space as hdf5 dataset using CanSAS file format

    Parameters:
    -----------
    :param path: Path where the data
    :type path: String
    :param filename: filename
    :type filename: String
    :param img: 2D array containing the intensity map in q-space
    :type img: ndarray
    :param qpar: 1D array containing the q horizontal/parallel coordinates
    :type qpar: ndarray
    :param qper: 1D array containing the q vertical/perpendicular coordinates
    :type qper: ndarray
    :param wav: wavelength in nm
    :type wav: float
    :param sdd: sample to detector distance in mm
    :type sdd: float
    :param beam_shape: beam shape in microns
    :type beam_shape: string
    '''

    stack = inspect.stack()
    function_name = stack[1][3]
    creator = function_name  # or program name

    full_filename = os.path.join(path, filename)

    with h5py.File(full_filename, "w") as nxroot:
        nxroot.attrs["creator"] = creator
        nxroot.attrs["default"] = "sasentry"
        nxroot.attrs["file_name"] = filename
        nxroot.attrs["file_time"] = datetime.datetime.now().isoformat()
        nxroot.attrs["HDF5_Version"] = h5py.version.hdf5_version
        nxroot.attrs["h5py_version"] = h5py.version.version
        nxroot.attrs["NeXus_version"] = "2020.1"

        nxentry = nxroot.create_group("sasentry")
        nxentry.attrs["NX_class"] = "NXentry"
        nxentry.attrs["canSAS_class"] = "SASentry"
        nxentry.attrs["version"] = "1.1"
        nxentry.attrs["default"] = "sasdata"

        nxentry.create_dataset("definition", data="NXcanSAS")
        nxentry.create_dataset("title", data="test_SMI_Nika")
        nxentry.create_dataset("run", data="test_SMI")

        nxdata = nxentry.create_group("sasdata")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["canSAS_class"] = "SASdata"
        nxdata.attrs["signal"] = "I"
        nxdata.attrs["I_axes"] = "Qy,Qx"
        nxdata.attrs["Qx_indices"] = 1  # for NXcanSAS
        nxdata.attrs["Qy_indices"] = 0  # for NXcanSAS

        ds = nxdata.create_dataset("I", data=img)
        ds.attrs["units"] = "arbitrary unit"

        ds = nxdata.create_dataset("Qx", data=qpar[::-1])
        ds.attrs["units"] = "1/A"
        nxdata.create_dataset("Qx_indices", data=[[1, ]])  # for NeXus (decided _after_ NXcanSAS)

        ds = nxdata.create_dataset("Qy", data=qver)
        ds.attrs["units"] = "1/A"
        nxdata.create_dataset("Qy_indices", data=[[0, ]])  # for NeXus (decided _after_ NXcanSAS)

        #         ds = nxdata.create_dataset("testx", data=idx_x)
        #         ds.attrs["units"] = "pixel"

        #         ds = nxdata.create_dataset("testy", data=idx_y)
        #         ds.attrs["units"] = "pixel"

        nxinstrument = nxentry.create_group("instrument")
        nxinstrument.attrs["NX_class"] = "NXinstrument"
        nxinstrument.attrs["canSAS_class"] = "SASinstrument"

        nxsource = nxinstrument.create_group("source")
        nxsource.attrs["NX_class"] = "NXsource"
        nxsource.attrs["canSAS_class"] = "SASsource"

        nxsource.create_dataset("radiation", data="Synchrotron X-ray Source")
        ds = nxsource.create_dataset("beam_shape", data=beam_shape)
        ds.attrs["units"] = "um2"
        ds = nxsource.create_dataset("wavelength", data=[wav])
        ds.attrs["units"] = "nm"

        nxdetector = nxinstrument.create_group("detector")
        nxdetector.attrs["NX_class"] = "NXdetector"
        nxdetector.attrs["canSAS_class"] = " SASdetector"

        nxdetector.create_dataset("name", data="Pilatus 300KW")
        ds = nxdetector.create_dataset("sdd", data=[sdd])
        ds.attrs["units"] = "mm"

    return nxdata