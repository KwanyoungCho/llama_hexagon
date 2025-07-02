#!/bin/bash
#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# Check if running as root"
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script with sudo"
    showHelp
    return 1
fi

# return back to the original directory if a command fails
# Since there are some commands using cd, this is necessary
original_dir=$(pwd)
cleanup() {
    # TODO: cleanup based on pwd
    cd "$original_dir"
    rm -rf .setup_tmp
}
trap cleanup EXIT;

showHelp() {
cat << EOF
Usage: source aimet_env_setup.sh --env-path <path> [--aimet-sdk-tar <sdk_dir>]
Creates a python virtual environment (if not already present) at the specified <path> and installs AIMET SDK along with necessary dependencies. If AIMET is installed in a directory other than the default QPM installation path, the user needs to specify it using --aimet-sdk-tar

-h, --help          Display help

--env-path          Specify the location where the virtual environment should be created or indicate the path to an existing virtual environment.

--aimet-sdk-tar    Specifies the location where the AIMET SDK installation files (.tar.gz) are stored.

EOF
}

# Ensure that the script is sourced and not directly executed
# This is important to set the "AIMET_ENV_PYTHON" in the environment variable
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "The script should not be executed directly"
    showHelp
    exit 1
fi

env_path=""
aimet_sdk_tar=""

options=$(getopt -l "help,env-path,aimet-sdk-tar:" -o "h")

#eval set -- "$options"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-path)
            env_path="$2"
            shift 2
            ;;
        --aimet-sdk-tar)
            aimet_sdk_tar="$2"
            shift 2
            ;;
        -h|--help)
            showHelp
            return 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1"
            showHelp
            return 1
    esac
done

if [[ -z "$env_path" ]]; then
    echo "Error: Please specify the env path using --env-path"
    showHelp
    return 1
fi

# Each QNN release is tied with a supported min and max AIMET version
# The script will choose the latest supported version that is installed
# If none of the supported versions are installed, then the script prompts the
# user to install a supported version and exits

MIN_VERSION="1.30" # Inclusive

verlte() {
    printf '%s\n' "$1" "$2" | sort -C -V
}

verlt() {
    ! verlte "$2" "$1"
}


aimet_torch_tarball=""
version=""
backend=""
is_valid_tar=true

if [ ! -z "$aimet_sdk_tar" ]; then
    aimet_sdk_tar=$(readlink -m "$aimet_sdk_tar")
    if [[ "$aimet_sdk_tar" =~ .*aimetpro-release-(.*)\.torch-(.*)-.*\.tar\.gz ]]; then
        version="${BASH_REMATCH[1]}"
        backend="${BASH_REMATCH[2]}"
        if [[ "$version" =~ ([0-9]+\.[0-9]+\.[0-9]+).* ]]; then
              version="${BASH_REMATCH[1]}"
        fi
        if verlt "$version" "$MIN_VERSION"; then
            is_valid_tar=false
            echo "Provided AIMET tarball version can't be used. Please provide AIMET version >= 1.30.0"
        fi
        if [ "$backend" != "cpu" ]  &&  [ "$backend" != "gpu" ]; then
            is_valid_tar=false
            echo "Provided backend is not supported by AIMET."
        fi
    else
        is_valid_tar=false
        echo "Provided AIMET tarball doesn't follow supported naming convention ( aimetpro-release-<version>.torch-<backend>-release.tar.gz)"
    fi

    if [ "$is_valid_tar" = "true" ] && [ -f "$aimet_sdk_tar" ]; then
        echo "AIMET version: $version, AIMET backend: $backend"
        aimet_torch_tarball="$aimet_sdk_tar"
    else
        echo "The specified tarball path $aimet_sdk_tar is not a valid."
        echo "Please provide a valid AIMET tarball path (aimetpro-release-*-torch-*-release.tar.gz)"
        return 1
    fi
fi


pick_aimet_version() {
    max_version=""
    for file in "$1"/*; do
        version=$(basename "$file")
        if ! verlt "$version" "$MIN_VERSION"; then
            if [ -z "$max_version" ] || verlt "$max_version" "$version" ; then
                max_version=$version
            fi
        fi
    done
    echo $max_version
}

# QPM installation conformance
# Assuming QNN root is the following path: /<qpm install dir>/aistack/qnn/<qnn version>/
# The script will be present in: /<qpm install dir>/aistack/qnn/<qnn version>/bin/
# Aimet tarball will be present in: /<qpm install dir>/aistack/aimet/<aimet version>
# Relative path to aimet traball dir: <Script source>/../../../aimet/<aimet version>/

# Absolute path of <qpm install dir>/aistack/qnn/<qnn version>/bin folder

if [ -z "$aimet_torch_tarball" ]; then

    script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

    # Go three levels up to <qpm install dir>/aistack
    aistack_root="$(dirname "$(dirname "$(dirname "$script_dir")")")"

    # Append "/aimet" to aistack root to get default AIMET installation directory
    aimet_root="${aistack_root}/aimet"

    echo "Checking for AIMET installation in "$aimet_root" "
    if [ ! -d "$aimet_root" ]; then
        echo "AIMET not be found in the default QPM installation directory ${aistack_root}"
        echo "Please install AIMET (version >= $MIN_VERSION ) using QPM or specify an alternate location using --aimet-sdk-tar <sdk_dir>"
        return
    else
        aimet_version=$(pick_aimet_version "$aimet_root")

        if [ -z "$aimet_version" ]; then
            echo "Unsupported AIMET version(s) installed. Please install AIMET (version >= $MIN_VERSION ) using QPM"
            return
        else
            qpm_aimet_sdk_tar="${aimet_root}/${aimet_version}"
            tarball=$(get_aimet_torch_tarball "$qpm_aimet_sdk_tar")
            if [ -z "$tarball" ]; then
                echo "Required AIMET SDK files(aimetpro-release-"$aimet_version"-*-torch-gpu-*.tar.gz or aimetpro-release-"$aimet_version"-*-torch-cpu) missing in "$qpm_aimet_sdk_tar""
                return
            fi
            aimet_torch_tarball="$tarball"
        fi
    fi
fi


echo "Using AIMET SDK: "$aimet_torch_tarball""


# At this point, "aimet_install_dir" contains the tarball files required for setting up the AIMET environment

env_path=$(readlink -m $env_path)

# Virtual environment where AIMET requirements will be installed
echo "Creating virtual environment at $env_path";

if [ -d "$env_path" ]; then
    echo "AIMET virtual environment already exists at $env_path"
    if ! source "$env_path/bin/activate"; then
        echo "Failed to activate virtual environment. Please delete the folder $env_path and re-run this script"
        return 1
    fi
else
    echo "Creating a new AIMET virtual environment..."
    if ! dpkg -s "python3-venv" &> /dev/null; then
        echo "python3-venv is not installed. This is needed to install and manage python dependencies for AIMET"
        sudo apt update -y
        if ! sudo apt install -y python3-venv; then
            "Could not install python3-venv"
            return 1
        fi
    fi
    # Create a virtual environment and activate it
    if python3 -m venv "$env_path"; then
        echo "Python virtual environment created successfully at $env_path";
        source "$env_path/bin/activate"
    else
        echo "Unable to create virtual environment: $env_path";
        return 1;
    fi
fi


pip3 install --upgrade pip

# Create a new temporary directory for setup artifacts
[ -d ".setup_tmp" ] && rm -rf ".setup_tmp"; mkdir ".setup_tmp"
cd ".setup_tmp"

#############################################################################################################
#TODO: Remove these filter utils once we split the reqs_deb_torch_gpu file in AIMET tarball dependencies

filter_cuda_toolkit_packages(){
   toolkit_packages=""
   if [ ! -z "$1" ]; then
       readarray -d ' ' -t packages <<< "$1"
       for package in ${packages[@]}; do
          readarray -d "=" -t array <<< "$package"
          if [ "${#array[@]}" = "1" ]; then
              toolkit_packages+=" $package"
          fi
       done
   fi
   echo $toolkit_packages
}

filter_additional_cuda_libraries(){
    cuda_libraries=""
    if [ ! -z "$1" ]; then
        readarray -d ' ' -t packages <<< "$1"
        for package in ${packages[@]}; do
           readarray -d "=" -t array <<< "$package"
           if [ "${#array[@]}" = "2" ]; then
              cuda_libraries+=" $package"
           fi
        done
    fi
    echo $cuda_libraries
}

################################################################################################################

get_missing_deb_packages () {
    all_packages=$(cat $1 | xargs -I{} echo "{}" | awk '{print $NF}')

    missing_deb_packages=""
    for package in $all_packages; do
        readarray -d "=" -t array <<< $package
        version=$(dpkg -s ${array[0]} | grep "^Version:" | tr -d 'Version:')
        if [ -z "$version" ]; then
            missing_deb_packages+=" $package"
        else
            if [ "${#array[@]}" = "2" ];  then
                if [ $version != ${array[1]} ]; then
                    missing_deb_packages+=" $package"
                fi
            fi
        fi
    done

    echo $missing_deb_packages
}

install_additional_cuda_libraries() {
    missing_cuda_libs=$1
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
    sudo apt-get update
    if ! sudo apt-get install $missing_cuda_libs; then
       echo "Failed to install missing additional CUDA based libraries!"
       return 1
    fi
}

get_cuda_version(){
   readarray -t packages < "dependencies/reqs_deb_torch_gpu.txt"
   sample_package=${packages[0]}
   readarray -d '-' -t array <<< "$sample_package"
   echo "${array[1]}.${array[2]}"
}

clean_up_earlier_cuda_installation_setup(){
   sudo apt-key del 7fa2af80
   if dpkg --list | grep cuda-keyring* >> /dev/null; then
      sudo apt-get purge --auto-remove "cuda-keyring*"
   fi
}

run_cuda_installation_setup_ubuntu() {
   missing_cuda_packages=$( get_missing_deb_packages "dependencies/reqs_deb_torch_gpu.txt" )
   if [ ! -z "$missing_cuda_packages" ]; then
       echo "Missing CUDA packages: $missing_cuda_packages"
       # Clean up earlier installation setup if present
       sudo apt-key del 7fa2af80
       # Install all missing CUDA dependencies from Ubuntu download index
       sudo apt-get update && sudo apt-get install -y gnupg2
       wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
       sudo dpkg -i cuda-keyring_1.0-1_all.deb
       sudo apt-get update
       if ! sudo apt-get install $missing_cuda_packages; then
           echo "Failed to install missing CUDA toolkit dependencies!"
           return 1
       fi
       # Add symbolic link for newly installed CUDA
       cuda_version=$( get_cuda_version )
       sudo rm -f /usr/local/cuda
       sudo ln -sf "/usr/local/cuda-$cuda_version" "/usr/local/cuda"
   fi
}

run_cuda_installation_setup_wsl() {
    # For WSL2, cuda toolkit packages needs to be installed from wsl-ubuntu download index to avoid
    # overwriting NVIDIA drivers, but the cuda libraries can only be installed from ubuntu index, as
    # they are common for both WSL and Ubuntu
    missing_cuda_packages=$( get_missing_deb_packages "dependencies/reqs_deb_torch_gpu.txt" )
    missing_cuda_toolkit_packages=$( filter_cuda_toolkit_packages "$missing_cuda_packages" )
    missing_cuda_libs=$( filter_additional_cuda_libraries "$missing_cuda_packages" )
    if [ ! -z "$missing_cuda_toolkit_packages" ] || [ ! -z "$missing_cuda_libs" ]; then
        # Clean up earlier installation setup if present
        clean_up_earlier_cuda_installation_setup
        sudo add-apt-repository -r "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
        sudo apt-get update && sudo apt-get install -y gnupg2
    fi
    if [ ! -z "$missing_cuda_toolkit_packages" ]; then
        # Install missing toolkit libraries according to WSL documentation for CUDA toolkit
        echo "Missing CUDA toolkit packages: $missing_cuda_toolkit_packages"
        wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        if ! sudo apt-get install $missing_cuda_toolkit_packages; then
            echo "Failed to install missing CUDA toolkit dependencies!"
            return 1
        fi
    fi
    if [ ! -z "$missing_cuda_libs" ]; then
       # Install missing non-toolkit libraries which needs to be installed from Ubuntu download index
       echo "Missing CUDA based libraries: $missing_cuda_libs"
       if ! install_additional_cuda_libraries "$missing_cuda_libs"; then
           return 1
       fi
    fi
    if [ ! -z "$missing_cuda_toolkit_packages" ] || [ ! -z "$missing_cuda_libs" ]; then
        # Add symbolic link for cuda-12.1
        cuda_version=$( get_cuda_version )
        sudo rm -f /usr/local/cuda
        sudo ln -sf "/usr/local/cuda-$cuda_version" "/usr/local/cuda"
    fi
}


install_aimet_sdk () {
    # Extract the tarball and install the dependencies
    PACKAGE=aimetpro-release
    mkdir $PACKAGE
    tar xzf $aimet_torch_tarball --strip-components 1 -C $PACKAGE
    cd $PACKAGE

    missing_deb_packages=$( get_missing_deb_packages "dependencies/reqs_deb_common.txt" )
    missing_deb_packages+=" $( get_missing_deb_packages "dependencies/reqs_deb_torch_common.txt" )"

    if [ -z $( echo $missing_deb_packages | tr -d ' ' ) ]; then
        missing_deb_packages=""
    fi

    if [ -n "$missing_deb_packages" ]; then
        echo "The following dependencies are missing and need to be installed:"$missing_deb_packages""
        echo "Installing the missing dependencies"
        sudo apt-get -y update
        if ! sudo apt-get -y install $missing_deb_packages; then
            echo "Failed to install missing dependencies"
            return 1
        fi
    else
        echo "All linux dependencies are already installed"
    fi

    if [ "$backend" = "gpu" ]; then
        echo "******************************************************************************************************"
        echo "CUDA version needed: $( get_cuda_version )"
        if [[ $(grep -i Microsoft /proc/version) ]]; then
            if run_cuda_installation_setup_wsl; then
               echo "All CUDA libraries are installed!"
            else
               echo "Couldn't install missing CUDA packages. Please install missing dependencies through NVIDIA CUDA documentation!"
            fi
        else
             if run_cuda_installation_setup_ubuntu; then
               echo "All CUDA libraries are installed!"
             else
               echo "Couldn't install missing CUDA packages. Please install missing dependencies through NVIDIA CUDA documentation!"
             fi
        fi
    fi
    echo "******************************************************************************************************"
    echo "Installing python dependencies..."

    TEMP_CACHE_DIR=`mktemp -d`
    if ! pip install pip/*.whl -f https://download.pytorch.org/whl/torch_stable.html --cache-dir $TEMP_CACHE_DIR; then
       echo "Could not install the AIMET SDK"
       return 1
    fi
}

if ! install_aimet_sdk; then
    echo "Failed to setup AIMET environment"
    cleanup
    return 1
fi

# HACK - module dataclasses has a bug. Can be removed safely
# https://github.com/huggingface/transformers/issues/8638
pip uninstall -y dataclasses

cd $original_dir # To the root
rm -rf .setup_tmp

# Use this environment variable in any program to reference the python executable for running AIMET-specific code
export AIMET_ENV_PYTHON=$env_path/bin/python;

if [ "$backend" = "gpu" ]; then
   export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
   export CUDA_TOOLKIT_PATH=/usr/local/cuda
   export CUDNN_INSTALL_PATH=/usr/local/cuda
   export CUDA_HOME=/usr/local/cuda
   export NVIDIA_DRIVER_CAPABILITIES=compute,utility
   export NVIDIA_VISIBLE_DEVICES=all
   export AIMET_VARIANT="torch-gpu"
   echo "CUDA specific environment variables are set! 'AIMET_VARIANT' set to "$AIMET_VARIANT""
else
   export AIMET_VARIANT="torch-cpu"
   echo "'AIMET_VARIANT' set to "$AIMET_VARIANT""
fi

echo "'AIMET_ENV_PYTHON' is set to "$AIMET_ENV_PYTHON" in this environment. Use this environment variable to
to reference the python executable for running AIMET specific code"

# Deactivate in the current context
deactivate;
