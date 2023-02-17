def natural_chr_sort(a, b):
    ac = a.replace("chr", "")
    ac = ac.split("_")[0]
    bc = b.replace("chr", "")
    bc = bc.split("_")[0]
    if bc.isnumeric() and ac.isnumeric() and bc != ac:
        if int(bc) < int(ac):
            return 1
        elif int(bc) > int(ac):
            return -1
        else:
            return 0
    else:
        if b < a:
            return 1
        elif a < b:
            return -1
        else:
            return 0
