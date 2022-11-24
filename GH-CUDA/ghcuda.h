#include "lib/constants.h"

void calculateIntersections(
                double *polyPX, double *polyPY, double *polyQX,  double *polyQY, 
                int sizeP, int sizeQ);
void calculateIntersections(
                coord_t *baseCoord, 
                coord_t *overlayCoords, 
                int sizeP, int sizeQ, double *cmbr,
                int *dbVNum, long *dbVPSNum, int *doVNum, long *doVPSNum, int bPID, int oPID,
                int *countNonDegenIntP, int *countNonDegenIntQ, 
                double **intersectionsP, double **intersectionsQ, int **alphaValuesP, int **alphaValuesQ,
                int **initLabelsP, int **initLabelsQ,
                int **neighborP, int **neighborQ);
void calculateIntersectionsMultipleComponents(
                double *polyPX, double *polyPY, 
                double *polyQX,  double *polyQY, 
                int sizeP, int sizeQ, int *sizesPP, int *sizesQQ, int sizePP, int sizeQQ,
                int *countNonDegenIntP, int *countNonDegenIntQ, 
                double **intersectionsP, double **intersectionsQ, int **alphaValuesP, int **alphaValuesQ,
                int **initLabelsP, int **initLabelsQ,
                int **neighborP, int **neighborQ);