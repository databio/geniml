import gzip
import os
import shutil
from logging import getLogger
from typing import List, NoReturn, Union

import requests
from ubiquerg import is_url

from .._version import __version__
from ..exceptions import GenimlBaseError
from ..io.io import BedSet, RegionSet
from ..io.utils import is_gzipped
from .const import (BEDFILE_URL_PATTERN, BEDSET_URL_PATTERN,
                    DEFAULT_BEDBASE_API, DEFAULT_BEDFILE_EXT,
                    DEFAULT_BEDFILE_SUBFOLDER, DEFAULT_BEDSET_EXT,
                    DEFAULT_BEDSET_SUBFOLDER, DEFAULT_CACHE_FOLDER,
                    MODULE_NAME)
from .utils import BedCacheManager, get_bbclient_path_folder

_LOGGER = getLogger(MODULE_NAME)


class BBClient(BedCacheManager):
    def __init__(
        self,
        cache_folder: str = DEFAULT_CACHE_FOLDER,
        bedbase_api: str = DEFAULT_BEDBASE_API,
    ):
        """
        BBClient to deal with download files from bedbase and caching them.

        :param cache_folder: path to local folder as cache of files from bedbase,
        if not given it will be the environment variable `BBCLIENT_CACHE`
        :param bedbase_api: url to bedbase
        """
        # get default cache folder from environment variable set by user
        super().__init__(get_bbclient_path_folder(cache_folder))
        self.bedbase_api = bedbase_api

    def load_bedset(self, bedset_id: str) -> BedSet:
        """
        Loads a BED set from cache, or downloads and caches it plus BED files in it if it doesn't exist

        :param bedset_id: unique identifier of BED set
        """

        file_path = self._bedset_path(bedset_id)

        if os.path.exists(file_path):
            _LOGGER.info(f"BED set {bedset_id} already exists in cache.")
            with open(file_path, "r") as file:
                extracted_data = file.readlines()
        else:
            extracted_data = self._download_bedset_data(bedset_id)
            # write the identifiers of BED files in the BedSet to a local .txt file
            with open(file_path, "w") as file:
                for value in extracted_data:
                    file.write(value + "\n")
            _LOGGER.info(f"BED set {bedset_id} downloaded and cached successfully.")

        # return the BedSet
        return BedSet(
            [self.load_bed(bedfile_id) for bedfile_id in extracted_data],
            identifier=bedset_id,
        )

    def _download_bedset_data(self, bedset_id: str) -> List[str]:
        """
        Download BED set from BEDbase API and return the list of identifiers of BED files in the set

        :param bedset_id: unique identifier of a BED set
        :return: the list of identifiers of BED files in the set
        """
        bedset_url = BEDSET_URL_PATTERN.format(bedbase_api=self.bedbase_api, bedset_id=bedset_id)
        response = requests.get(bedset_url)
        data = response.json()
        extracted_data = [entry.get("record_identifier") for entry in data["bedfile_metadata"]]

        return extracted_data

    def load_bed(self, bed_id: str) -> RegionSet:
        """
        Loads a BED file from cache, or downloads and caches it if it doesn't exist

        :param bed_id: unique identifier of a BED file
        """
        file_path = self._bedfile_path(bed_id)

        if os.path.exists(file_path):
            _LOGGER.info(f"BED file {bed_id} already exists in cache.")
        else:
            file_path = self._bedfile_path(bed_id)

            if os.path.exists(file_path):
                _LOGGER.info(f"BED file {bed_id} already exists in cache.")
            # if not in the cache, download from BEDbase and write to file in cache
            else:
                bed_data = self._download_bed_file_from_bb(bed_id)
                with open(file_path, "wb") as f:
                    f.write(bed_data)
                _LOGGER.info(f"BED file {bed_id} was downloaded and cached successfully")

        return RegionSet(regions=file_path)

    def add_bedset_to_cache(self, bedset: BedSet) -> str:
        """
        Add a BED set to the cache

        :param bedset: the BED set to be added, a BedSet class
        :return: the identifier if the BedSet object
        """
        bedset_id = bedset.compute_bedset_identifier()
        file_path = self._bedset_path(bedset_id)
        if os.path.exists(file_path):
            _LOGGER.info(f"{file_path} already exists in cache.")
        else:
            with open(file_path, "w") as file:
                for bedfile in bedset:
                    bedfile_id = self.add_bed_to_cache(bedfile)
                    file.write(bedfile_id + "\n")
        return bedset_id

    def add_bed_to_cache(self, bedfile: Union[RegionSet, str]) -> str:
        """
        Add a BED file to the cache

        :param bedfile: a RegionSet class or a path to a BED file to be added to cache
        :return: the identifier if the BedFile object
        """
        if isinstance(bedfile, str):
            bedfile = RegionSet(bedfile)
        elif not isinstance(bedfile, RegionSet):
            raise TypeError(
                f"Input must be a RegionSet or a path to a BED file, not {type(bedfile)}"
            )

        bedfile_id = bedfile.compute_bed_identifier()
        file_path = self._bedfile_path(bedfile_id)
        if os.path.exists(file_path):
            _LOGGER.info(f"{file_path} already exists in cache.")
        else:
            if bedfile.path is None or is_url(bedfile.path):
                # write the regions to .bed.gz file
                with gzip.open(file_path, "wt") as f:
                    for region in bedfile:
                        f.write(f"{region.chr}\t{region.start}\t{region.end}\n")

            else:
                # copy the BED file out of cache
                if is_gzipped(bedfile.path):
                    shutil.copyfile(bedfile.path, file_path)
                else:
                    # https://docs.python.org/3/library/gzip.html
                    with open(bedfile.path, "rb") as f_in:
                        with gzip.open(file_path, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

        return bedfile_id

    def seek(self, identifier: str) -> str:
        """
        Get local path to BED file or BED set with specific identifier

        :param identifier: the unique identifier
        :return: the local path of the file
        """

        # check if any BED set has that identifier
        file_path = self._bedset_path(identifier, False)
        if os.path.exists(file_path):
            return file_path
        else:
            file_path = self._bedfile_path(identifier, False)
            if os.path.exists(file_path):
                return file_path
            else:
                raise FileNotFoundError(f"{identifier} does not exist in cache.")

    def remove_bedset_from_cache(self, bedset_id: str, remove_bed_files: bool = False) -> NoReturn:
        """
        Remove a BED set from cache

        :param bedset_id: the identifier of BED set
        :param remove_bed_files: whether also remove BED files in the BED set
        :raise FileNotFoundError: if the BED set does not exist in cache
        """

        file_path = self.seek(bedset_id)
        if remove_bed_files:
            with open(file_path, "r") as file:
                extracted_data = file.readlines()
            for bedfile_id in extracted_data:
                self.remove_bedfile_from_cache(bedfile_id)

        self._remove(file_path)

    def _download_bed_file_from_bb(self, bedfile: str) -> bytes:
        """
        Download BED file from BEDbase API and return the file content as bytes

        :param bedfile: unique identifier of a BED file
        :return: the file content as bytes
        """

        bed_url = BEDFILE_URL_PATTERN.format(bedbase_api=self.bedbase_api, bed_id=bedfile)
        response = requests.get(bed_url)
        response.raise_for_status()
        return response.content

    def _bedset_path(self, bedset_id: str, create: bool = True) -> str:
        """
        Get the path of a BED set's .txt file with given identifier

        :param bedset_id: the identifier of BED set
        :param create: whether the cache path needs creating
        :return: the path to the .txt file
        """

        subfolder_name = DEFAULT_BEDSET_SUBFOLDER
        file_extension = DEFAULT_BEDSET_EXT

        return self._cache_path(bedset_id, subfolder_name, file_extension, create)

    def _bedfile_path(self, bedfile_id: str, create: bool = True) -> str:
        """
        Get the path of a BED file's .bed.gz file with given identifier

        :param bedfile_id: the identifier of BED file
        :param create: whether the cache path needs creating
        :return: the path to the .bed.gz file
        """

        subfolder_name = DEFAULT_BEDFILE_SUBFOLDER
        file_extension = DEFAULT_BEDFILE_EXT

        return self._cache_path(bedfile_id, subfolder_name, file_extension, create)

    def _cache_path(
        self, identifier: str, subfolder_name: str, file_extension: str, create: bool = True
    ) -> str:
        """
        Get the path of a file in cache folder

        :param identifier: the identifier of BED set or BED file
        :param subfolder_name: "bedsets" or "bedfiles"
        :param file_extension: ".txt" or ".bed.gz"
        :param create: whether the cache path needs creating
        :return: the path to the file
        """

        filename = f"{identifier}{file_extension}"
        folder_name = os.path.join(self.cache_folder, subfolder_name, identifier[0], identifier[1])
        if create:
            self.create_cache_folder(folder_name)
        return os.path.join(folder_name, filename)

    def remove_bedfile_from_cache(self, bedfile_id: str) -> NoReturn:
        """
        Remove a BED file from cache

        :param bedfile_id: the identifier of BED file
        :raise FileNotFoundError: if the BED set does not exist in cache
        """

        file_path = self.seek(bedfile_id)
        self._remove(file_path)

    @staticmethod
    def _remove(file_path: str) -> NoReturn:
        """
        Remove a file within the cache with given path, and remove empty subfolders after removal
        Structure of folders in cache:
        cache_folder
            bedfiles
                a/b/ab1234xyz.bed.gz
            bedsets
                c/d/cd123hij.txt

        :param file_path: the path to the file
        :return: NoReturn
        """
        # the subfolder that matches the second digit of the identifier
        sub_folder_2 = os.path.split(file_path)[0]
        # the subfolder that matches the first digit of the identifier
        sub_folder_1 = os.path.split(sub_folder_2)[0]

        os.remove(file_path)

        # if the subfolders are empty after removal, remove the folders too
        if len(os.listdir(sub_folder_2)) == 0:
            os.rmdir(sub_folder_2)
            if len(os.listdir(sub_folder_1)) == 0:
                os.rmdir(sub_folder_1)
