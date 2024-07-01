1. Download `LA_ALL_2018` from https://anl.app.box.com/s/traooe2ovuwuliaphpj6qxkimghcqqew/folder/154954796640. It should be a zip file. Put it in this dir.
2. Run `setup.sh`
3. Go into `pbs_submit_manager.sh` and change the queue/walltime
4. Run `pbs_submit_manager.sh`
5. Change line 18 in `submit.sh` to point to your current dir. Rn it points to `/eagle/projects/radix-io/sockerman/Dask-Parallel-Traffic-Benchmark/methods/dask_pipeline/` which will not help.
6. Run `./pbs_submit_manger.sh`  