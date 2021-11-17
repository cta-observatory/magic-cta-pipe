#!/bin/bash

obs_ids_array=(
5582 5583 5584 5630 5631 5632 5633 5634 5635 5636 5637 5638 5639 5696 5697 5698 5699 5700 5701 5702 5703 5704 5719 5722 5723 5724 5725 5726 5742 5743 5744 5745 5746 5747 5748 5891 5892 5912 5913 5914 5915 5916
)

for obs_id in ${obs_ids_array[@]}; do

    obs_id=`printf %05d ${obs_id}`
    echo "Run${obs_id}"

    #files_list=`ls /fefs/aswg/data/real/DL1/*/v0.6.3_v05/dl1_LST-1.Run${obs_id}.*.h5`
    files_list=`ls /fefs/aswg/data/real/DL1/*/v0.7.*/tailcut84/dl1_LST-1.Run${obs_id}.*.h5`
    
    if [ -n "${files_list}" ]; then

        echo -e "${files_list}"

        for file in ${files_list}; do

            date=`echo ${file} | awk -F'/' '{print $7}'`

            if [[ ${date} =~ ^([0-9]{4})([0-9]{2})([0-9]{2})$ ]]; then

                year=${BASH_REMATCH[1]}
                month=${BASH_REMATCH[2]}
                day=${BASH_REMATCH[3]}

            fi

            #output_dir=/home/yoshiki.ohtani/combined_analysis/real/RSOph/data_check/check_pointing/data/${year}_${month}_${day}/LST-1
            output_dir=/home/yoshiki.ohtani/combined_analysis/real/RSOph/${year}_${month}_${day}/2.event_coincidence/data/dl1/LST-1
	        mkdir -p ${output_dir}

            ln -sf ${file} ${output_dir}/

        done

        echo -e "--> ${output_dir}\n"

    else
        echo ""

    fi

done

