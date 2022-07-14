#!/bin/bash

set -e

echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/calibrated/20210314_M1_05095172.001_Y_CrabNebula-W0.40+035.root" >  test_data_real.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/calibrated/20210314_M1_05095172.002_Y_CrabNebula-W0.40+035.root" >> test_data_real.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/calibrated/20210314_M2_05095172.001_Y_CrabNebula-W0.40+035.root" >> test_data_real.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/calibrated/20210314_M2_05095172.002_Y_CrabNebula-W0.40+035.root" >> test_data_real.txt

echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/images/20210314_M1_05095172.001_I_CrabNebula-W0.40+035.h5" >  test_data_images_real.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/images/20210314_M1_05095172.002_I_CrabNebula-W0.40+035.h5" >> test_data_images_real.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/images/20210314_M2_05095172.001_I_CrabNebula-W0.40+035.h5" >> test_data_images_real.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/images/20210314_M2_05095172.002_I_CrabNebula-W0.40+035.h5" >> test_data_images_real.txt

echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/real/superstar/20210314_05095172_S_CrabNebula-W0.40+035.root" > test_data_superstar_real.txt

echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/calibrated/GA_M1_za35to50_8_824318_Y_w0.root" >  test_data_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/calibrated/GA_M1_za35to50_8_824319_Y_w0.root" >> test_data_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/calibrated/GA_M2_za35to50_8_824318_Y_w0.root" >> test_data_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/calibrated/GA_M2_za35to50_8_824319_Y_w0.root" >> test_data_simulated.txt

echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/images/GA_M1_za35to50_8_824318_I_w0.h5" >  test_data_images_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/images/GA_M1_za35to50_8_824319_I_w0.h5" >> test_data_images_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/images/GA_M2_za35to50_8_824318_I_w0.h5" >> test_data_images_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/images/GA_M2_za35to50_8_824319_I_w0.h5" >> test_data_images_simulated.txt

echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/superstar/GA_za35to50_8_824318_S_w0.root" > test_data_superstar_simulated.txt
echo "https://webdav-magic.pic.es:8451/Users/ctapipe_io_magic/test_data/simulated/superstar/GA_za35to50_8_824319_S_w0.root" > test_data_superstar_simulated.txt

if [ -z "$TEST_DATA_USER" ]; then
    echo -n "Username: "
    read TEST_DATA_USER
    echo
fi

if [ -z "$TEST_DATA_PASSWORD" ]; then
    echo -n "Password: "
    read -s TEST_DATA_PASSWORD
    echo
fi

if ! wget \
    -i test_data_real.txt \
    --user="$TEST_DATA_USER" \
    --password="$TEST_DATA_PASSWORD" \
    --no-check-certificate \
    --no-verbose \
    --timestamping \
    --directory-prefix=test_data/real/calibrated; then
    echo "Problem in downloading the test data set for real data."
fi

if ! wget \
    -i test_data_images_real.txt \
    --user="$TEST_DATA_USER" \
    --password="$TEST_DATA_PASSWORD" \
    --no-check-certificate \
    --no-verbose \
    --timestamping \
    --directory-prefix=test_data/real/images; then
    echo "Problem in downloading the test data set for real data."
fi

if ! wget \
    -i test_data_superstar_real.txt \
    --user="$TEST_DATA_USER" \
    --password="$TEST_DATA_PASSWORD" \
    --no-check-certificate \
    --no-verbose \
    --timestamping \
    --directory-prefix=test_data/real/superstar; then
    echo "Problem in downloading the test data set for real data."
fi

if ! wget \
    -i test_data_simulated.txt \
    --user="$TEST_DATA_USER" \
    --password="$TEST_DATA_PASSWORD" \
    --no-check-certificate \
    --no-verbose \
    --timestamping \
    --directory-prefix=test_data/simulated/calibrated; then
    echo "Problem in downloading the test data set for simulated data."
fi

if ! wget \
    -i test_data_images_simulated.txt \
    --user="$TEST_DATA_USER" \
    --password="$TEST_DATA_PASSWORD" \
    --no-check-certificate \
    --no-verbose \
    --timestamping \
    --directory-prefix=test_data/simulated/images; then
    echo "Problem in downloading the test data set for simulated data."
fi

if ! wget \
    -i test_data_superstar_simulated.txt \
    --user="$TEST_DATA_USER" \
    --password="$TEST_DATA_PASSWORD" \
    --no-check-certificate \
    --no-verbose \
    --timestamping \
    --directory-prefix=test_data/simulated/superstar; then
    echo "Problem in downloading the test data set for simulated data."
fi

rm -f test_data_real.txt test_data_simulated.txt test_data_images_real.txt test_data_superstar_real.txt test_data_images_simulated.txt test_data_superstar_simulated.txt
