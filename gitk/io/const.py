from typing import Literal

MAF_HUGO_SYMBOL_COL_NAME = "Hugo_Symbol"
MAF_ENTREZ_GENE_ID_COL_NAME = "Entrez_Gene_Id"
MAF_CENTER_COL_NAME = "Center"
MAF_NCBI_BUILD_COL_NAME = "NCBI_Build"
MAF_CHROMOSOME_COL_NAME = "Chromosome"
MAF_START_COL_NAME = "Start_position"
MAF_END_COL_NAME = "End_position"
MAF_STRAND_COL_NAME = "Strand"

MAF_COLUMN = Literal[
    "Hugo_Symbol",
    "Entrez_Gene_Id",
    "Center",
    "NCBI_Build",
    "Chromosome",
    "Start_Position",
    "End_Position",
    "Strand",
]

MAF_FILE_DELIM = "\t"
