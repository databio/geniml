import os

MODULE_NAME = "bbclient"

DEFAULT_BEDBASE_API = os.getenv("BEDBASE_API") or "https://api.bedbase.org"
DEFAULT_BEDSET_SUBFOLDER = "bedsets"
DEFAULT_BEDFILE_SUBFOLDER = "bedfiles"
DEFAULT_BEDSET_EXT = ".txt"
DEFAULT_BEDFILE_EXT = ".bed.gz"

BEDFILE_URL_PATTERN = "{bedbase_api}/v1/objects/bed.{bed_id}.bed_file/access/http/bytes"
BEDSET_URL_PATTERN = "{bedbase_api}/v1/bedset/{bedset_id}/bedfiles"
BED_TOKENS_PATTERN = "{bedbase_api}/v1/bed/{bed_id}/tokens/{universe_id}/info"

BBCLIENT_CACHE_ENV = "BBCLIENT_CACHE"

HOME_PATH = os.getenv("HOME")
if not HOME_PATH:
    HOME_PATH = os.path.expanduser("~")
DEFAULT_CACHE_FOLDER = os.getenv(BBCLIENT_CACHE_ENV) or os.path.join(HOME_PATH, ".bbcache/")

DEFAULT_ZARR_FOLDER = "tokens.zarr"
DEFAULT_BUCKET_NAME = "bedbase"
DEFAULT_BUCKET_FOLDER = "bed_files"
