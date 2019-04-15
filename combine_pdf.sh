pdftk $(ls */*z-inf_inf.pdf --ignore=oldsubmissions/*) cat output Confusion_Matrices_z_all.pdf
pdftk $(ls */*z-inf_0.4.pdf --ignore=oldsubmissions/*) cat output Confusion_Matrices_z_lt_0p4.pdf
pdftk $(ls */*z0.4_inf.pdf --ignore=oldsubmissions/*) cat output Confusion_Matrices_z_gt_0p4.pdf

