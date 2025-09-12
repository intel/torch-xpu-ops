#!/bin/bash
set -xe

if [ "$1" == "lts2" ];then
	# Clean
	rm -rf .intel ./intel /opt/intel .cache .config .condarc .conda .gitconfig
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
	rm -rf /etc/apt/sources.list.d/*
	apt-get autoclean && apt-get clean
	apt update

	# Install common
	apt install gpg-agent wget curl sudo gcc g++ gcc-11 g++-11 cmake git unzip zip libgl1 zlib1g-dev gh expect numactl tmux htop -y
	update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 && \
		update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 11 && \
		update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-11 11 && \
		gcc -v && g++ -v && gcov -v

	# Install LTS2
	. /etc/os-release
	rm -rf /etc/apt/sources.list.d/*
	wget -qO - https://repositories.intel.com/gpu/intel-graphics.key |gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
	echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu ${VERSION_CODENAME}/lts/2523 unified" |tee /etc/apt/sources.list.d/intel-gpu-${VERSION_CODENAME}.list
	rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
	apt-get autoclean && apt-get clean
	apt update

	apt install -y \
		linux-headers-$(uname -r) \
		linux-modules-extra-$(uname -r) \
		flex bison \
		intel-fw-gpu intel-i915-dkms xpu-smi
	apt install -y \
		intel-opencl-icd libze-intel-gpu1 libze1 \
		intel-media-va-driver-non-free libmfx-gen1 libvpl2 \
		libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
		libglapi-mesa libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
		mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo intel-ocloc
	apt install -y \
		libigc-dev intel-igc-cm libigdfcl-dev libigfxcmrt-dev libze-dev

    # Upgrade to latest
    apt upgrade -y
    apt autoremove -y
fi
