#!/bin/bash

# This script runs all the modules in one fell swooop
# Change the module values to TRUE/FALSE to run or skip specific modules
# This is for a local caller, not for a cluster environment


DOWNLOAD=FALSE
PREPROCESS=FALSE
ILLUMINATION_CORRECTION=FALSE
SEGMENT=TRUE
EXTRACT_FEATURES=TRUE
IMAGE_BASED_PROFILING=TRUE

##################################
# Place holder for data download / streaming
####################################
if [ "$DOWNLOAD" = TRUE ] ; then
    echo "Running Data Download Module..."
    # bash 0.data_download/run_data_download.sh
    echo "Data Download Module is not yet implemented."
else
    echo "Skipping Data Download Module..."
fi

##################################
# Preprocessing
# Maximum intensity projection
##################################
if [ "$PREPROCESS" = TRUE ] ; then
    echo "Running Preprocessing Module..."
    cd 1.max_projection || exit
    # shellcheck disable=SC1091
    source check_and_max_proj.sh
    cd .. || exit
else
    echo "Skipping Preprocessing Module..."
fi

####################################
# Illumination correction
####################################
if [ "$ILLUMINATION_CORRECTION" = TRUE ] ; then
    echo "Running Illumination Correction Module..."
    cd 2.illumination_correction || exit
    # shellcheck disable=SC1091
    source run_ic.sh
    cd .. || exit
else
    echo "Skipping Illumination Correction Module..."
fi

##################################
# Cell Segmentation
##################################
if [ "$SEGMENT" = TRUE ] ; then
    echo "Running Cell Segmentation Module..."
    cd 3.cell_segmentation || exit
    # shellcheck disable=SC1091
    source run_segmentation.sh
    cd .. || exit
else
    echo "Skipping Cell Segmentation Module..."
fi

##################################
# Feature Extraction
##################################
if [ "$EXTRACT_FEATURES" = TRUE ] ; then
    echo "Running Feature Extraction Module..."
    cd 3.feature_extraction || exit
    # shellcheck disable=SC1091
    source run_local_featurization.sh
    cd .. || exit
else
    echo "Skipping Feature Extraction Module..."
fi


##################################
# Image-based Profiling
##################################
if [ "$IMAGE_BASED_PROFILING" = TRUE ] ; then
    echo "Running Image-based Profiling Module..."
    cd 4.preprocess_features || exit
    # shellcheck disable=SC1091
    source run_preprocessing.sh
    cd .. || exit
else
    echo "Skipping Image-based Profiling Module..."
fi
