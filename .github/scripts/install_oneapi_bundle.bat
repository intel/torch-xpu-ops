set ONEAPI_PARENT_DIR=C:\Program Files (x86)\Intel
set INTEL_ONEAPI_PYTORCH_BUNDLE_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/5ca2021d-dd1a-4ab1-bd52-758fe63cf827/w_intel-for-pytorch-gpu-dev_p_0.5.2.19_offline.exe
set INTEL_ONEAPI_PYTORCH_BUNDLE_DISPLAY_NAME=intel.oneapi.win.intel-for-pytorch-gpu-dev.product
set INSTALL_FRESH_ONEAPI=1

if not exist "%ONEAPI_PARENT_DIR%\oneAPI" (
    set INSTALL_FRESH_ONEAPI=1
)


if "%INSTALL_FRESH_ONEAPI%"=="1" (
    IF EXIST "%ONEAPI_PARENT_DIR%\oneAPI\Installer\installer.exe" (
        "%ONEAPI_PARENT_DIR%\oneAPI\Installer\installer.exe" --list-products > oneapi_products_before_uninstall.log
        IF EXIST tmp_oneapi_uninstall_version.log (
            del tmp_oneapi_uninstall_version.log
        )
    
        "%ONEAPI_PARENT_DIR%\oneAPI\Installer\installer.exe" --list-products > tmp_oneapi_uninstall_version.log
        for /f "tokens=1,2" %%a in (tmp_oneapi_uninstall_version.log) do (
            if "%%a"=="%INTEL_ONEAPI_PYTORCH_BUNDLE_DISPLAY_NAME%" (
                echo Version: %%b
                start /wait "Installer Title" "%ONEAPI_PARENT_DIR%\oneAPI\Installer\installer.exe" --action=remove --eula=accept --silent --product-id %INTEL_ONEAPI_PYTORCH_BUNDLE_DISPLAY_NAME% --product-ver %%b --log-dir uninstall_bundle
            )
        )
    
        IF EXIST tmp_oneapi_uninstall_version.log (
            del tmp_oneapi_uninstall_version.log
        )
        if errorlevel 1 exit /b
        if not errorlevel 0 exit /b
    )

    curl -o oneapi_bundle.exe --retry 3 --retry-all-errors -k %INTEL_ONEAPI_PYTORCH_BUNDLE_URL%
    start /wait "Intel Pytorch Bundle Installer" "oneapi_bundle.exe" --action=install --eula=accept --silent --log-dir install_bundle
    if errorlevel 1 exit /b
    if not errorlevel 0 exit /b
)
