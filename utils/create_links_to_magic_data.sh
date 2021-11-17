#!/bin/bash

obs_ids_array=(
5099070 5099071 5099101 5099102 5099103 5099104 5099105 5099106 5099107 5099108 5099109 5099110 5099168 5099169 5099170 5099171 5099172 5099173 5099174 5099175 5099176 5099477 5099478 5099507 5099508 5099509 5099510 5099511
)

for obs_id in ${obs_ids_array[@]}; do

    obs_id=`printf %08d ${obs_id}`
    echo "obs_id = ${obs_id}"

    #files_list=`ls /home/yoshiki.ohtani/Data/MAGIC/RSOph/SuperStar/*/*_${obs_id}_S_*.root`
    files_list=`ls /fefs/onsite/common/MAGIC/data/M*/event/Calibrated/*/*/*/*_${obs_id}.*_Y_*.root`

    if [ -n "${files_list}" ]; then

        echo "${files_list}"

        for file in ${files_list}; do

            date_magic=`echo ${file} | awk -F'/' '{print $NF}' | awk -F'_' '{print $1}'`

            if [[ ${date_magic} =~ ^([0-9]{4})([0-9]{2})([0-9]{2})$ ]]; then

                year=${BASH_REMATCH[1]}
                month=${BASH_REMATCH[2]}
                day=${BASH_REMATCH[3]}

            fi

            date_lst=`date -d "${year}/${month}/${day} yesterday" "+%Y_%m_%d"`

            #output_dir=/home/yoshiki.ohtani/combined_analysis/real/RSOph/data_check/check_data_quality/data/${date_lst}
            output_dir=/home/yoshiki.ohtani/combined_analysis/real/RSOph/${date_lst}/1.magic_cal_to_dl1/data/calibrated/run${obs_id}
            mkdir -p ${output_dir}

            ln -sf ${file} ${output_dir}/

        done

        echo -e "--> ${output_dir}\n"

    else
        echo ""

    fi

done
