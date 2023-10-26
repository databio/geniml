# DEFAULT_BEDBASE_API = "https://api.bedbase.org"
DEFAULT_BEDBASE_API = "http://bedbase.org/api"
DEFAULT_BEDSET_SUBFOLDER = "bedsets"
DEFAULT_BEDFILE_SUBFOLDER = "bedfiles"
DEFAULT_BEDSET_EXT = ".txt"
DEFAULT_BEDFILE_EXT = ".bed.gz"


BEDSET_URL_PATTERN = "{bedbase_api}/bedset/{bedset_id}/bedfiles?ids=md5sum"
BEDFILE_URL_PATTERN = "{bedbase_api}/bed/{bedfile_id}/file/bedfile"
