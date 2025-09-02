# VS Code Setup for YOLO Processing

## 🚀 Quick Start Options

You have **3 ways** to run YOLO processing through VS Code:

### **Option 1: VS Code + GitHub Codespaces (Recommended)**
- **Best performance** (32GB RAM cloud processing)
- **PDFs stay on your Mac** (uploaded temporarily)
- **Fastest processing** (~20-30 minutes for all files)

### **Option 2: VS Code + Local Mac**
- **Everything stays local** on your Mac
- **Slower processing** (~2-3 hours for all files)
- **Reduced memory usage** (optimized for local)

### **Option 3: VS Code + Remote Codespaces**
- **Best of both worlds** - VS Code interface + cloud power
- **Full debugging capabilities**
- **Real-time monitoring**

---

## 🔧 Setup Instructions

### **Prerequisites**

1. **Install VS Code Extensions:**
   ```
   - Python (ms-python.python)
   - GitHub Codespaces (GitHub.codespaces)
   - Remote Development (ms-vscode-remote.vscode-remote-extensionpack)
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ **Option 1: VS Code + GitHub Codespaces**

### **Step 1: Open in Codespaces**
1. In VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: `Codespaces: Open Repository in Codespace`
3. Select your repository
4. **Important**: Choose **32GB RAM** machine type

### **Step 2: Run Processing**
1. Open the integrated terminal in VS Code
2. Run the test first:
   ```bash
   python LightweightTest.py
   ```
3. If tests pass, run YOLO processing:
   ```bash
   python yoloprocessing.py
   ```

### **Step 3: Monitor Progress**
- Watch real-time progress in the terminal
- Check the **Problems** panel for any issues
- Monitor resource usage in the Codespaces panel

### **Step 4: Download Results**
- Results are automatically saved to `yolo_processed_data/`
- Download the entire folder when complete
- PDFs remain on your local machine

---

## 🖥️ **Option 2: VS Code + Local Mac**

### **Step 1: Open Project in VS Code**
1. Open VS Code
2. File → Open Folder → Select your project folder
3. Wait for Python extension to activate

### **Step 2: Run Local Processing**
1. Open the integrated terminal (`Ctrl+`` ` or `Cmd+`` `)
2. Run the local-optimized version:
   ```bash
   python yoloprocessing_local.py
   ```

### **Step 3: Monitor Local Performance**
- Watch memory usage in Activity Monitor
- Check CPU usage in VS Code's status bar
- Monitor progress in the terminal

### **Performance Expectations (Local Mac):**
- **Processing Time**: 2-3 hours for all 57 files
- **Memory Usage**: ~8-12GB RAM
- **Image Quality**: 2x (moderate quality)
- **Workers**: 4 parallel processes

---

## 🔧 **Option 3: VS Code + Remote Codespaces**

### **Step 1: Connect to Codespace**
1. In VS Code, press `Ctrl+Shift+P`
2. Type: `Remote-SSH: Connect to Host`
3. Select your Codespace from the list

### **Step 2: Use VS Code Debugging**
1. Go to **Run and Debug** panel (`Ctrl+Shift+D`)
2. Select **"Run YOLO Processing (Codespaces)"**
3. Click the green play button

### **Step 3: Debug Features**
- Set breakpoints in the code
- Step through processing
- Monitor variables in real-time
- Use the debug console

---

## 🎯 **VS Code Launch Configurations**

The project includes pre-configured launch configurations:

### **Available Configurations:**
1. **"Run YOLO Processing (Local Mac)"** - Local processing
2. **"Run YOLO Processing (Codespaces)"** - Cloud processing
3. **"Run Lightweight Test"** - System verification
4. **"Debug YOLO Processing"** - Debug mode

### **How to Use:**
1. Press `F5` or go to **Run and Debug** panel
2. Select the configuration from the dropdown
3. Click the green play button

---

## 📊 **Monitoring & Debugging**

### **Real-time Monitoring:**
- **Terminal Output**: Live progress updates
- **Problems Panel**: Error detection
- **Output Panel**: Detailed logging
- **Status Bar**: Python interpreter info

### **Debugging Features:**
- **Breakpoints**: Set breakpoints in code
- **Variable Inspection**: Hover over variables
- **Call Stack**: Track function calls
- **Watch Expressions**: Monitor specific values

### **Performance Monitoring:**
- **Memory Usage**: Check in Activity Monitor
- **CPU Usage**: Monitor in VS Code status bar
- **Disk Usage**: Watch output folder size

---

## 🛠️ **Troubleshooting**

### **Common VS Code Issues:**

1. **Python Interpreter Not Found:**
   ```bash
   # In VS Code terminal:
   which python
   # Then set in VS Code: Ctrl+Shift+P → "Python: Select Interpreter"
   ```

2. **Extension Not Working:**
   - Reload VS Code: `Ctrl+Shift+P` → "Developer: Reload Window"
   - Reinstall extension if needed

3. **Memory Issues (Local):**
   - Close other applications
   - Reduce workers in `yoloprocessing_local.py`
   - Monitor Activity Monitor

4. **Codespaces Connection Issues:**
   - Check internet connection
   - Verify GitHub account permissions
   - Try different machine type

### **Performance Tips:**

1. **For Local Processing:**
   - Close unnecessary VS Code tabs
   - Disable unused extensions
   - Monitor system resources

2. **For Codespaces:**
   - Use 32GB RAM machine
   - Keep VS Code focused on this project
   - Monitor Codespaces usage

---

## 📁 **Output Structure**

### **Local Processing:**
```
yolo_processed_data_local/
├── images/           # PNG images (2x quality)
├── metadata/         # PDF metadata
├── features/         # Extracted features
└── reports/          # Processing reports
```

### **Codespaces Processing:**
```
yolo_processed_data/
├── images/           # PNG images (4x ultra-high quality)
├── metadata/         # PDF metadata
├── features/         # Enhanced features
└── reports/          # Detailed reports
```

---

## 🎉 **Success Indicators**

### **Local Processing Complete:**
```
🎉 YOLO Processing Complete!
  ✅ Processed: 57
  ❌ Failed: 0
  ⏱️  Total Time: 2.5 hours
  📊 Average: 158.2 seconds per file
  🚀 Success Rate: 100.0%
```

### **Codespaces Processing Complete:**
```
🎉 YOLO Processing Complete!
  ✅ Processed: 57
  ❌ Failed: 0
  ⏱️  Total Time: 25.3 minutes
  📊 Average: 26.6 seconds per file
  🚀 Success Rate: 100.0%
```

---

## 📞 **Support**

### **VS Code Issues:**
1. Check VS Code documentation
2. Verify extension installations
3. Check Python interpreter settings

### **Processing Issues:**
1. Review log files in output directories
2. Check system resources
3. Verify PDF files are accessible

### **Codespaces Issues:**
1. Check GitHub account permissions
2. Verify machine type selection
3. Monitor Codespaces usage limits

---

**Happy Processing with VS Code! 🚀**
