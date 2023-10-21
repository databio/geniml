import gzip
import os
import shutil
from typing import List

import requests

from ..io import is_gzipped
from ..io.io import BedSet, RegionSet
from .const import (
    BEDFILE_URL_PATTERN,
    BEDSET_URL_PATTERN,
    DEFAULT_BEDBASE_API,
    DEFAULT_BEDFILE_EXT,
    DEFAULT_BEDFILE_SUBFOLDER,
    DEFAULT_BEDSET_EXT,
    DEFAULT_BEDSET_SUBFOLDER,
)
from .utils import BedCacheManager


class BBClient(BedCacheManager):
    def __init__(self, cache_folder: str, bedbase_api: str = DEFAULT_BEDBASE_API):
        """
        BBClient to deal with download files from bedbase and caching them.

        :param cache_folder: path to local folder as cache of files from bedbase
        :param bedbase_api: url to bedbase
        """
        super().__init__(cache_folder)
        self.bedbase_api = bedbase_api

    def _download_bed_data(self, bedfile_id: str) -> bytes:
        """
        Download BED file from BEDbase API and return the file content as bytes

        :param bedfile_id: unique identifier of a BED file
        """
        bed_url = BEDFILE_URL_PATTERN.format(bedbase_api=self.bedbase_api, bedfile_id=bedfile_id)
        response = requests.get(bed_url)
        response.raise_for_status()

        return response.content

    def load_bedset(self, bedset_id: str) -> BedSet:
        """
        Loads a BED set from cache, or downloads and caches it plus BED files in it if it doesn't exist

        :param bedset_id: unique identifier of BED set
        """

        file_path = self._bedset_path(bedset_id)

        if os.path.exists(file_path):
            print(f"BED set {bedset_id} already exists in cache.")
            with open(file_path, "r") as file:
                extracted_data = file.readlines()
        # if the BedSet is not in cache, download it from BEDBase
        else:
            extracted_data = self._download_bedset_data(bedset_id)
            # write the identifiers of BED files in the BedSet to a local .txt file
            with open(file_path, "w") as file:
                for value in extracted_data:
                    file.write(value + "\n")
            print(f"BED set {bedset_id} downloaded and cached successfully.")

        # return the BedSet
        return BedSet(
            [self.load_bed(bedfile_id) for bedfile_id in extracted_data],
            identifier=bedset_id,
        )

    def _download_bedset_data(self, bedset_id: str) -> List[str]:
        """
        Download BED set from BEDbase API and return the list of identifiers of BED files in the set

        :param bedset_id: unique identifier of a BED set
        """
        bedset_url = BEDSET_URL_PATTERN.format(bedbase_api=self.bedbase_api, bedset_id=bedset_id)
        response = requests.get(bedset_url)
        data = response.json()
        extracted_data = [entry[0] for entry in data["data"]]

        return extracted_data

    def load_bed(self, bedfile_id: str) -> RegionSet:
        """
        Loads a BED file from cache, or downloads and caches it if it doesn't exist

        :param bedfile_id: unique identifier of a BED file
        """
        # the path of the .txt file of the BED set
        file_path = self._bedfile_path(bedfile_id)

        if os.path.exists(file_path):
            print(f"BED file {bedfile_id} already exists in cache.")
        # if not in the cache, download from BEDbase and write to file in cache
        else:
            bed_data = self._download_bed_data(bedfile_id)
            with open(file_path, "wb") as f:
                f.write(bed_data)
            print("File downloaded and cached successfully.")

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
            print(f"{file_path} already exists in cache.")
        else:
            with open(file_path, "w") as file:
                for bedfile in bedset:
                    bedfile_id = self.add_bed_to_cache(bedfile)
                    file.write(bedfile_id + "\n")
        return bedset_id

    def add_bed_to_cache(self, bedfile: RegionSet) -> str:
        """
        Add a BED file to the cache

        :param bedfile: the BED file to be added, a BedFile class
        :return: the identifier if the BedFile object
        """

        # bedfile_id = bedfile.compute_bed_identifier()
        bedfile_id = bedfile.compute_bed_identifier()
        file_path = self._bedfile_path(bedfile_id)
        if os.path.exists(file_path):
            print(f"{file_path} already exists in cache.")
        else:
            if bedfile.path is None:
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
        file_path = self._bedset_path(identifier)
        if os.path.exists(file_path):
            return file_path
        # check if any BED file has that identifier
        else:
            file_path = self._bedfile_path(identifier)
            if os.path.exists(file_path):
                return file_path
            # return message if the id is not cached
            else:
                return f"{identifier} does not exist in cache."

    def _bedset_path(self, bedset_id: str) -> str:
        """
        Get the path of a BED set's .txt file with given identifier

        :param bedset_id: the identifier of BED set
        :return: the path to the .txt file
        """

        subfolder_name = DEFAULT_BEDSET_SUBFOLDER
        file_extension = DEFAULT_BEDSET_EXT

        return self._cache_path(bedset_id, subfolder_name, file_extension)

    def _bedfile_path(self, bedfile_id: str) -> str:
        """
        Get the path of a BED file's .bed.gz file with given identifier

        :param bedfile_id: the identifier of BED set
        :return: the path to the .bed.gz file
        """

        subfolder_name = DEFAULT_BEDFILE_SUBFOLDER
        file_extension = DEFAULT_BEDFILE_EXT

        return self._cache_path(bedfile_id, subfolder_name, file_extension)

    def _cache_path(self, identifier: str, subfolder_name: str, file_extension: str) -> str:
        """
        Get the path of a file in cache folder

        :param identifier: the identifier of BED set or BED file
        :param subfolder_name: "bedsets" or "bedfiles"
        :param file_extension: ".txt" or ".bed.gz"
        :return: the path to the file
        """
        filename = f"{identifier}{file_extension}"
        folder_name = os.path.join(self.cache_folder, subfolder_name, identifier[0], identifier[1])

        self.create_cache_folder(folder_name)
        return os.path.join(folder_name, filename)

    def remove_bedfile_from_cache(self, bedfile_id: str):
        """
        Remove a BED file from cache

        :param bedfile_id: the identifier of BED file
        """
        file_path = self.seek(bedfile_id)
        if "does not" in file_path:
            print(file_path)
        else:
            self._remove(file_path)

    def remove_bedset_from_cache(self, bedset_id: str, remove_bed_files: bool = False):
        """
        Remove a BED set from cache

        :param bedset_id: the identifier of BED set
        :param remove_bed_files: whether also remove BED files in the BED set
        """
        file_path = self.seek(bedset_id)
        if "does not" in file_path:
            print(file_path)
        else:
            if remove_bed_files:
                with open(file_path, "r") as file:
                    extracted_data = file.readlines()
                for bedfile_id in extracted_data:
                    self.remove_bed_files(bedfile_id)

            self._remove(file_path)

    def _remove(self, file_path: str):
        """
        Remove a file within the cache with given path, and remove empty subfolders after removal
        Structure of folders in cache:
        cache_folder
            bedfiles
                a/b/ab1234xyz.bed.gz
            ..
            bedsets
                c/d/cd123hij.txt

        :param file_path: the path to the file
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
