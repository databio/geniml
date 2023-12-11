# Add Random Regions Only in Valid Regions

Using the basic `--add` option, regions are added randomly onto any chromosome at any location, without any regard for non-coding regions. For use cases of Bedshift more rooted in biology, this effect is not desirable. The `--add-valid` option gives the user the ability to specify a BED file indicating areas where it is valid to add regions. Thus, if an `--add-valid` file has only coding regions, then regions will be randomly added only in those areas. Here is an example:

```
bedshift -b mydata.bed -a 0.5 --add-valid coding.bed --addmean 500 --addstdev 200
```

`coding.bed` contains large regions of the genome which are coding. Added regions can be anywhere inside of those regions. In addition, the method considers the size of the valid regions in deciding where the new regions will be added, so the smaller valid regions will contain proportionally less new regions than the larger valid regions.