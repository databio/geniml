import os
import shlex
import subprocess
import tempfile
from subprocess import PIPE, Popen


def prep_data(folder, file, tmp_file):
    """File sort and merge"""
    fin = os.path.join(folder, file)
    stdin_1 = None
    if file.endswith(".gz"):
        command0 = f"zcat {fin}"
        fin = ""
        p0 = Popen(shlex.split(command0), stdout=PIPE)
        stdin_1 = p0.stdout
    command1 = f"sort -k1,1V -k2,2n {fin}"
    command2 = "bedtools merge"
    p1 = Popen(shlex.split(command1), stdin=stdin_1, stdout=PIPE)
    p2 = Popen(shlex.split(command2), stdin=p1.stdout, stdout=PIPE)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    output = p2.communicate()[0]
    tmp_file.write(output)
    tmp_file.seek(0)


def check_if_uni_sorted(universe):
    """Check if regions in file are sorted"""
    command0 = f"sort -k1,1V -k2,2n -c {universe}"
    output = subprocess.run(shlex.split(command0))
    if output.returncode:
        raise Exception("Universe not sorted")


def check_if_uni_flexible(universe):
    with open(universe) as u:
        l = u.readline()
        l = l.split("\t")
        if len(l) < 6:
            raise Exception("Universe is not flexible")


def process_line(line):
    """Helper for reading in bed file line"""
    line = line.split("\t")[:3]
    chrom = line[0]
    pos = [int(i) for i in line[1:]]
    start = pos[0]
    return pos, start, chrom


def chrom_cmp_bigger(a, b):
    """Natural check if chromosomes name is bigger"""
    ac = a.replace("chr", "")
    ac = ac.split("_")[0]
    bc = b.replace("chr", "")
    bc = bc.split("_")[0]
    if bc.isnumeric() and ac.isnumeric() and bc != ac:
        if int(bc) < int(ac):
            return True
        else:
            return False
    else:
        if b < a:
            return True
        else:
            return False


def process_db_line(dn, pos_index):
    """Helper for reading in universe bed file line"""
    dn = dn.split("\t")
    region_chrom = dn[0]
    return [int(dn[p]) for p in pos_index], region_chrom
