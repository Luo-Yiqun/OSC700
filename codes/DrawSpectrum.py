import numpy as np
import json


class Absorption():
    def __init__(
            self,
            photon_energy,
            extinction_coefficient, 
            optical_gap : float = None
            ):
        r"""
        @author: Yiqun Luo, luo2@andrew.cmu.edu
        Initialize an Absorption object to handle and visualize absorption spectra.

        This class integrates Siyu's old version and some new functions.
        It can now handle with .json input, which is the PAH101 standard form,
        and .dat, which is the BerkeleyGW output file by using class methods,
        as well as the solar spectrum .txt file.

        Input:
        photon_energy: with unit eV
        extinction_coefficient:
        optical_gap: with unit eV

        Example:
        >>> absorp = Absorption.from_json(r"path/to/HBZCOR.json")
        >>> absorp = Absorption.extinction_coefficient_dat(r"path/to/absorption_eh_a.dat")
        >>> absorp = Absorption.from_dat(
        >>>    r"path/to/absorption_eh_a.dat",
        >>>    r"path/to/absorption_eh_b.dat",
        >>>    r"path/to/absorption_eh_c.dat"
        >>>    )
        >>> solar_spectrum = absorp.sl_read(r"path/to/allsmrtetr.txt")

        >>> t1 = absorp.absorbance('a', normalize = True)
        >>> t2 = absorp.absorbance('b', normalize = True)
        >>> t3 = absorp.absorbance('c', normalize = True)
        >>> t = absorp.average_absorbance(normalize = True)

        >>> fig, ax = plt.subplots()
        >>> ax.plot(t1[0], t1[1], '-b', label = 'a')
        >>> ax.plot(t2[0], t2[1], '-g', label = 'b')
        >>> ax.plot(t3[0], t3[1], '-m', label = 'c')
        >>> ax.plot(t[0], t[1], '-k', label = 'tot', linewidth = 4)
        >>> ax.axvline(absorp.opt_gap, linestyle = "--", label = "Optical gap")
        >>> ax2 = ax.twinx()
        >>> ax2.plot(solar_spectrum[0], solar_spectrum[1], label = "Solar spectrum")
        >>> ax.legend()
        >>> ax2.legend()
        >>> plt.show()
        """
        self.energy = photon_energy
        self.extinct_coef = extinction_coefficient
        self.opt_gap = optical_gap
    
    @staticmethod
    def sl_read(file : str): 
        """
        Reads spectral data from a file, assuming data starts from the second line.
        Returns the wavelength and modified spectral intensity.
        """
        try:
            with open(file) as f:
                lines = f.readlines()[1:2002]
            wavelength = [1240.0 / float(line.split()[0]) for line in lines]
            intensity = [float(line.split()[1]) * float(line.split()[0]) / (1240.0 / float(line.split()[0])) for line in lines]
            return np.flipud(wavelength), np.flipud(intensity)
        except Exception as e:
            print(f"Failed to read or process the file {file}: {e}")
            return
        
    @staticmethod
    def approximate_extinction_coefficient(epsilon_1, epsilon_2):
        """
        This approximate function is rarely used and kept for historical reasons.
        """
        return epsilon_2 / 2
    
    @staticmethod
    def exact_extinction_coefficient(epsilon_1, epsilon_2):
        return np.sqrt((np.sqrt(epsilon_1 ** 2 + epsilon_2 ** 2) - epsilon_1) / 2)
    
    @classmethod
    def extinction_coefficient_dat(cls, file : str, exact : bool = True):
        extinct_coef_func = cls.exact_extinction_coefficient if exact else cls.approximate_extinction_coefficient
        with open(file) as f:
            data = np.array([i.split() for i in f.readlines()[4:]], dtype = float)
        return data[:, 0], extinct_coef_func(data[:, 2], data[:, 1])

    @classmethod
    def from_dat(cls, file_a : str, file_b : str = None, file_c : str = None, exact : bool = True):
        """
        Instantiate an object using BerkeleyGW output .dat file.
        This classmethod allows only input one direction.
        """
        energy, extinct_coef = {}, {}
        if file_a:
            energy['a'], extinct_coef['a'] = cls.extinction_coefficient_dat(file_a, exact)
        if file_b:
            energy['b'], extinct_coef['b'] = cls.extinction_coefficient_dat(file_b, exact)
        if file_c:
            energy['c'], extinct_coef['c'] = cls.extinction_coefficient_dat(file_c, exact)
        return cls(energy, extinct_coef)
    
    @classmethod
    def from_json(cls, file : str, exact : bool = True):
        """
        Instantiate an object using the PAH101 standard .json file.
        """
        with open(file) as f:
            data = json.load(f)["gwbse"]
        opt_gap = data["bse_Es"]
        data = data["absorption"]
        energy, extinct_coef = {}, {}
        energy['a'] = np.array(data['a'])[:, 0]
        energy['b'] = np.array(data['b'])[:, 0]
        energy['c'] = np.array(data['c'])[:, 0]
        extinct_coef_func = cls.exact_extinction_coefficient if exact else cls.approximate_extinction_coefficient
        extinct_coef['a'] = extinct_coef_func(np.array(data['a'])[:, 2], np.array(data['a'])[:, 1])
        extinct_coef['b'] = extinct_coef_func(np.array(data['b'])[:, 2], np.array(data['b'])[:, 1])
        extinct_coef['c'] = extinct_coef_func(np.array(data['c'])[:, 2], np.array(data['c'])[:, 1])
        return cls(energy, extinct_coef, opt_gap)
    
    def absorption_coefficient(self, direction = 'a', lim = None, normalize = False):
        """
        Input:
        direction:
        lim: energy range of interest with unit eV.
            If none, return all the energy range
        normalize: Whether to normalize the spectrum for comparison convenience

        Out:
        Absorption coefficient with unit cm^-1
        """
        assert direction in self.energy, "Direction invalid or not read!"
        energy = self.energy[direction]
        out = 4 * np.pi / 1240.0E-7 * energy * self.extinct_coef[direction]
        if lim:
            idx = (energy > lim[0]) & (energy < lim[1])
            energy = energy[idx]
            out = out[idx]
        if normalize:
            out /= np.max(out)
        return energy, out
    
    def average_absorption_coefficient(self, lim = None, normalize = False):
        """
        Input:
        lim: energy range of interest with unit eV.
            If none, return all the energy range
        normalize: Whether to normalize the spectrum for comparison convenience

        Out:
        Average absorption coefficient with unit cm^-1

        This method uses the simple arithmatic average for convenience.
        """
        out = np.mean((
            self.absorption_coefficient('a', lim),
            self.absorption_coefficient('b', lim),
            self.absorption_coefficient('c', lim)
        ), axis = 0
        )
        energy, out = out.reshape(2, -1)
        if normalize:
            out /= np.max(out)
        return energy, out

    def absorbance(self, direction = 'a', thickness : float = 5e-8, lim = None, normalize = False):
        """
        Input:
        direction:
        thickness: thickness of the material with unit m
        lim: energy range of interest with unit eV.
            If none, return all the energy range
        normalize: Whether to normalize the spectrum for comparison convenience

        out:
        Absorbance
        """
        energy, out = self.absorption_coefficient(direction, lim)
        out = out * thickness * 100
        if normalize:
            out /= np.max(out)
        return energy, out
    
    def average_absorbance(self, thickness : float = 5e-8, lim = None, normalize = False):
        """
        Input:
        thickness: thickness of the material with unit m
        lim: energy range of interest with unit eV.
            If none, return all the energy range
        normalize: Whether to normalize the spectrum for comparison convenience

        out:
        Average absorbance

        This method uses the simple arithmatic average for convenience.
        """
        out = np.mean((
            self.absorbance('a', thickness, lim),
            self.absorbance('b', thickness, lim),
            self.absorbance('c', thickness, lim),
        ), axis = 0
        )
        energy, out = out.reshape(2, -1)
        if normalize:
            out /= np.max(out)
        return energy, out

    def intensity(self, direction = 'a', thickness : float = 5e-8, lim = None, normalize = False):
        """
        Input:
        direction:
        thickness: thickness of the material with unit m
        lim: energy range of interest with unit eV.
            If none, return all the energy range
        normalize: Whether to normalize the spectrum for comparison convenience

        out:
        Intensity
        """
        energy, out = self.absorbance(direction, thickness, lim)
        out = np.exp(-out)
        if normalize:
            out /= np.max(out)
        return energy, out
    
    def average_intensity(self, thickness : float = 5e-8, lim = None, normalize = False):
        """
        Input:
        thickness: thickness of the material with unit m
        lim: energy range of interest with unit eV.
            If none, return all the energy range
        normalize: Whether to normalize the spectrum for comparison convenience

        out:
        Average intensity

        This method uses the simple arithmatic average for convenience.
        """
        out = np.mean((
            self.intensity('a', thickness, lim),
            self.intensity('b', thickness, lim),
            self.intensity('c', thickness, lim),
        ), axis = 0
        )
        energy, out = out.reshape(2, -1)
        if normalize:
            out /= np.max(out)
        return energy, out