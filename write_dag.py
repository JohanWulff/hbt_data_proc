import json
from glob import glob
import os
from argparse import ArgumentParser
from subprocess import Popen, PIPE
from pathlib2 import Path


def make_parser():
    parser = ArgumentParser(description="Submit predicting of LLR \
Samples")
    parser.add_argument("-s", "--submit_base", type=str, 
                        help="Base dir to submit from")
    parser.add_argument("-o", "--output_dir" ,type=str,
                        help="Dir to write output files to")
    parser.add_argument("-y", "--year", type=str, 
                        help="16, 17 or 18")
    parser.add_argument("--add_htautau", action="store_true",
                        help="If set, evaluate htautau model.")
    parser.add_argument("-j", "--json", type=str,
                        help="JSON File containing paths to samples")
    parser.add_argument("-n", "--n_files" ,type=int, default=5,
                        help="number of files per job")
    return parser
                            

def checkmake_dir(path):
    if not os.path.exists(path):
        print(f"{path} does not exist.")
        print("Shall I create it now?")
        yn = input("[y/n] ?")
        if yn.strip().lower() == 'y':
            print('Creating dir(s)!')
            os.makedirs(path)
        else:
            raise ValueError(f"{path} does not exist")


def return_subfile(outdir, executable):
    file_str = f"executable={executable}\n\
log                     = singularity.$(ClusterId).log\n\
error                   = singularity.$(ClusterId).$(ProcId).err\n\
output                  = singularity.$(ClusterId).$(ProcId).out\n\
\n\
should_transfer_files = YES\n\
MY.JobFlavour = \"microcentury\"\n\
output_destination      = root://eosuser.cern.ch//{outdir}\n\
MY.XRDCP_CREATE_DIR     = True\n\
MY.SingularityImage     = \"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/jwulff/lumin_3.8:latest\"\n\
\n\
Arguments = $(FILES)\n\
queue"
# transfer_input_files    = root://eosuser.cern.ch//eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/branchnames.py, root://eosuser.cern.ch//eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/feature_calc.py,root://eosuser.cern.ch//eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/features.py,root://eosuser.cern.ch//eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/first_nn.py,root://eosuser.cern.ch//eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/helpers.py\n\
    return file_str


def return_executable(sum_w, year, sample_id, add_htautau):
    function_call_str = f'python ${{EXE}} -f $@ -s {sum_w} -i {sample_id} -y {year}'
    if add_htautau:
        function_call_str += " --add_htautau"
    file_str = f'#!/usr/bin/bash\n\
EXE="/eos/user/j/jowulff/res_HH/hbt_data_proc/evaluate_klub.py"\n\
{function_call_str} || exit 1\n\
exit 0'
    return file_str


def parse_goodfile_txt(goodfile:Path,):
    skims_dir = goodfile.absolute().parent
    with open(goodfile) as gfile:
        gfiles = sorted([Path(line.rstrip()) for line in gfile])
        if len(gfiles) == 0:
            print(f"Found 0 files in {goodfile}. Globbing all .root files in skim dir.")
            # goodfiles.txt is empty: just glob all .root files in 
            # skims dir and hope they're good
            gfiles = sorted([i for i in skims_dir.glob("*.root")])
        else:
            # check if the paths have been updated
            if gfiles[0].parent != skims_dir:
                # if not stick the filename on the end of the provided path
                # and hope for the best
                gfiles = [skims_dir / i.name for i in gfiles]
    return [str(gfile) for gfile in gfiles]


def main(submit_base_dir: str, 
         sample_json: str, 
         outdir: str,
         year: str,
         add_htautau: bool,
         n_files: int=5):

    with open(sample_json) as f:
        d = json.load(f)
        # select the year
        d = d[year]

    if not submit_base_dir.startswith("/afs"):
        raise ValueError("Submission must happen from /afs!")
    checkmake_dir(submit_base_dir)
    checkmake_dir(outdir)
    
    for i, sample in enumerate(d):
        if "_VBF_" in sample:
            continue
        print(f"Creating submission dir and writing dag \
files for sample ({i+1}/{len(d)})\r", end="")
        # data samples are channel-dependant
        submit_dir = submit_base_dir.rstrip("/")+f"/{sample}"
        eos_outdir = outdir.strip("/")+f"/{sample}"
        #xrdcp_create_dir = True in submit script
        if not os.path.exists(submit_dir):
            os.mkdir(submit_dir)
        dagfile = submit_dir+f"/{sample}.dag"
        submitfile = submit_dir+f"/{sample}.submit"
        afs_exe = submit_dir+f"/executable.sh"
        path = d[sample]["Path"]
        sum_w = d[sample]["Sum_w"]
        sample_id = d[sample]["Sample_ID"]
        goodfile = path+"/goodfiles.txt"
        if not os.path.exists(afs_exe):
            with open(afs_exe, "w") as f:
                print(return_executable(sum_w, year, sample_id, add_htautau), file=f)
            prcs = Popen(f"chmod 744 {afs_exe}",shell=True, 
                        stdin=PIPE, stdout=PIPE, encoding='utf-8')
            out, err = prcs.communicate()
            if err:
                print(err)
                raise ValueError(f"Unable to chmod {afs_exe} to 744")
        else:
            print(f"\n {afs_exe} already exists.. Not creating new one \n")
        if not os.path.exists(goodfile):
            print(f"{sample} does not have a goodfile.txt at \
{path}")
            gfiles = glob(d[sample]["Path"]+"/*.root")
        else:
            gfiles = parse_goodfile_txt(Path(goodfile))
        # filter files for broken files
        filechunks = [gfiles[i:i+n_files] for i in range(0, len(gfiles), n_files)]
        if not os.path.exists(dagfile):
            with open(dagfile, "x") as dfile:
                for chunk in filechunks:
                    print(f"JOB {chunk[0].split('/')[-1]} {submitfile}", file=dfile)
                    print(f'VARS {chunk[0].split("/")[-1]} FILES="{" ".join(chunk)}"', file=dfile)
                #for file in gfiles:
                    #print(f"JOB {file.split('/')[-1]} {submitfile}", file=dfile)
                    #print(f'VARS {file.split("/")[-1]} FILES="{file}"', file=dfile)
            submit_string = return_subfile(outdir=outdir, executable=afs_exe)
        else:
            print(f"\n {dagfile} already exists.. Not creating new one \n")

        if not os.path.exists(submitfile):
            submit_string = return_subfile(outdir=eos_outdir, executable=afs_exe)
            with open(submitfile, "x") as subfile:
                print(submit_string, file=subfile)
        else:
            print(f"\n {submitfile} already exists.. Not creating new one \n")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(submit_base_dir=args.submit_base,
         sample_json=args.json,
         outdir=args.output_dir,
         year=args.year,
         add_htautau=args.add_htautau,
         n_files=args.n_files)
