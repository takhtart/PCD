# Setup Instructions

## Requirements:
- **Visual Studio 2022**
- **PCL Library 1.14.1+**
- **CMake**
- **Kinect for Windows SDK 2.0**

---

## 1. Download Visual Studio 2022 (Community Edition)

1. Navigate to [Visual Studio Download](https://visualstudio.microsoft.com/downloads/)
2. Download and run the `VisualStudioSetup.exe`
3. During the setup, under the **Desktop & Mobile** category, check the option for **Desktop development with C++**
4. Complete the installation of Visual Studio 2022

---

## 2. Install PCL Library

Follow the guide to install PCL: [PCL Installation Guide](https://github.com/PointCloudLibrary/pcl/issues/4462)

> **Note:** The version of PCL being installed is **PCL 1.14.1** (latest release). CMake compilation should be done using **Visual Studio 17 2022**. Other setup and installation tests should remain the same.

### Updated Environment Variables:
- **PCL_ROOT**
  - Value: `C:\Program Files\PCL 1.14.1`

- **Path**
  - Add the following paths to the system's PATH variable:
    - `C:\Program Files\OpenNI2`
    - `C:\Program Files\PCL 1.14.1\bin`
    - `C:\Program Files\PCL 1.14.1\3rdParty`
    - `C:\Program Files\PCL 1.14.1\3rdParty\VTK\bin`
    - `C:\Program Files\OpenNI2\Tools`

- **OPENNI2_INCLUDE64**
  - Value: `C:\Program Files\OpenNI2\Include\`

- **OPENNI2_LIB64**
  - Value: `C:\Program Files\OpenNI2\Lib\`

- **OPENNI2_REDIST64**
  - Value: `C:\Program Files\OpenNI2\Redist\`

---

## 2. Install OpenCV Library

> **Note:** The version of OpenCV being installed is **OpenCV 4.11.0** (latest release).
Link to Github Releases Page: [OpenCV Releases](https://github.com/opencv/opencv/releases)

### Installation:
- Download the `opencv-4.11.0-windows.exe` under Assets

- Run the exe, Under **Extract to** set path to be `C:\`
- Press `Extract`, window will close automatically upon extraction
- Verify that the folder exists at path `C:\opencv`

### Add the following User Environment Variables:
- **OpenCV_DIR**
  - Value: `C:\opencv\build\`

- **Path**
  - Add the following paths to the system's PATH variable:
    - `C:\opencv\build\x64\vc16\bin`
    - `C:\opencv\build\x64\vc16\lib`

---

## 3. Install Kinect for Windows SDK 2.0

1. Download and install the Kinect for Windows SDK 2.0 from the following link:  
   [Kinect for Windows SDK 2.0 Download](https://www.microsoft.com/en-ca/download/details.aspx?id=44561)

2. **Test Sensor Connection:**
   - Open the installed **SDK Browser 2.0 (Kinect For Windows)** application
   - Run the **Kinect Configuration Verifier** to ensure the Kinect sensor is properly connected to your PC

---

## 4. Compile and Run `cloud_viewer.cpp`

1. Navigate to the **src** folder of the GitHub PCD Repository
2. Compile and run the `cloud_viewer.cpp` file to test the setup

> Ensure all dependencies are correctly configured and the Kinect sensor is properly connected before running the program.
