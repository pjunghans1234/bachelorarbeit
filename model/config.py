from pathlib import Path


path_to_online_data = Path('/net/atmos/data/cmip6-ng')

path_to_local_data = Path('/home/pjunghans/BachelorArbeit/Play_data_MPI-ESM1-2-LR/cmip6-ng-coarse-grid')


path_to_parametere_output = Path('/home/pjunghans/BachelorArbeit/model_output/calibrated_parameters')

path_to_scores_output = Path('/home/pjunghans/BachelorArbeit/model_output/score_ds')

path_to_plot_output = Path('/home/pjunghans/BachelorArbeit/model_output/plots')



run = 'r1i1p1f1'

#1-30 sind safe, danach gibt es bei "mrsol" fehlende Datensätze
all_runs = ["r50i1p1f1","r1i1p1f1","r2i1p1f1","r3i1p1f1","r4i1p1f1","r5i1p1f1","r6i1p1f1","r7i1p1f1","r8i1p1f1","r9i1p1f1",
            "r10i1p1f1","r11i1p1f1","r12i1p1f1","r13i1p1f1","r14i1p1f1","r15i1p1f1","r16i1p1f1","r17i1p1f1","r18i1p1f1","r19i1p1f1",
            "r20i1p1f1","r21i1p1f1","r22i1p1f1","r23i1p1f1","r24i1p1f1","r25i1p1f1","r26i1p1f1","r27i1p1f1","r28i1p1f1","r29i1p1f1",
            "r30i1p1f1","r31i1p1f1","r32i1p1f1","r33i1p1f1","r34i1p1f1","r35i1p1f1","r36i1p1f1","r37i1p1f1","r38i1p1f1","r39i1p1f1",
            "r40i1p1f1","r41i1p1f1","r42i1p1f1","r43i1p1f1","r44i1p1f1","r45i1p1f1","r46i1p1f1","r47i1p1f1","r48i1p1f1","r49i1p1f1"]