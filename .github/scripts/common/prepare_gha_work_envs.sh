set -x
echo gha_runner: $gha_runner

# if [ -z ${gha_runner} ]; then
#     echo "Need to specify gha_runner: Borealis or SDP?"
#     exit 1
# elif [ ${gha_runner} != 'a21-surrogate4-libo' ] && [ ${gha_runner} != 'lnl' ] && [ ${gha_runner} != 'mtl' ] && [ ${gha_runner} != 'bmg' ] && [ ${gha_runner} != 'arc' ]; then
#     echo "Currently we only supports running on lnl/mtl/bmg/arc"
#     exit 1
# fi 

if [ -z ${WORK_WEEK} ]; then
    echo "Need to specify WORK_WEEK: This is the folder name for storing wheels and logs"
    exit 1   
fi

if [ -z ${pytorch_branch_commit} ]; then
    echo "Need to specify pytorch_branch_commit in yml file: such as 2.15.0"
    exit 1   
fi


if [ -z ${python_version} ]; then
    echo "Need to specify python_version: "
    echo "Such as 3.7, 3.8, 3.9, 3.10, 3.11 for general python; which will be used by "
    echo "conda create -n ${CONDA_ENV} python=${python_version} -y  "
    echo "Such as 3.10 for python"
    exit 1   
fi

if [ -z ${workspace_path} ]; then
    # by default uses huxue's path
    workspace_path='C:\libohao' 
fi

# if [ ${gha_runner} = 'lnl' ] && [ -z $fwk_version ] ; then
#     echo "Need to specify fwk_version on SDP: "
#     echo "Such as itex_xpu or itex_gpu"
#     exit 1
# fi





# if [ ${gha_runner} = 'a21-surrogate4-libo' ]; then
#     work_dir=${workspace_path}/logs/${WORK_WEEK}/${test_type}_${pt_version}_${python_version}
# elif [ ${gha_runner} = 'lnl' ] && [ ${gha_runner} = 'mtl' ] && [ ${gha_runner} = 'bmg' ] && [ ${gha_runner} = 'arc' ];; then    
#     work_dir=${workspace_path}/logs/${WORK_WEEK}/${test_type}_${pt_version}_${python_version}
#     whl_path=${workspace_path}/whls/${WORK_WEEK}/${test_type}_${pt_version}_${python_version}  
#     CONDA_ENV=${WORK_WEEK}_${test_type}_${pt_version}_${python_version}  
# fi
work_dir=${workspace_path}\logs\${WORK_WEEK}\${test_type}_${pt_version}_${python_version}
whl_path=${workspace_path}\whls\${WORK_WEEK}\${test_type}_${pt_version}_${python_version}  
CONDA_ENV=${WORK_WEEK}_${test_type}_${pt_version}_${python_version}  

#rm -rf ${work_dir}
# mkdir -p ${work_dir} 
# mkdir -p $whl_path 

# cp_version="cp$(echo ${python_version}  | sed -r 's/\.//')"
version_info_file=${work_dir}/version_info.log



# Writes all the infos to env variables for later use in another job
echo "echo CONDA_ENV=${CONDA_ENV} >> \$GITHUB_ENV" >> env_var_config.sh  
echo "echo workspace_path=$workspace_path >> \$GITHUB_ENV" >> env_var_config.sh     
echo "echo work_dir=${work_dir} >> \$GITHUB_ENV" >> env_var_config.sh
echo "echo version_info_file=$version_info_file >> \$GITHUB_ENV" >> env_var_config.sh

