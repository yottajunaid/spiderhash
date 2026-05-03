<div align="center">
  <img src="src/assets/logo.png" alt="SpiderHash Logo" width="220">

  # SpiderHash

  ## **SpiderHash** is a GUI-based Hash Cracker utility written in Python. It supports cracking over **22 hashing algorithms**, offering both the convenience of a graphical user interface and the power of configurable wordlists / bruteforce behavior. 
  
  <br>

  <a href="https://github.com/user-attachments/files/27314192/spiderhash_report.pdf">
    <img src="https://img.shields.io/badge/Project_Report-blue?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xNCAySDZjLTEuMSAwLTEuOTkuOS0xLjk5IDJMNCAyMGMwIDEuMS44OSAyIDEuOTkgMkgxOGMxLjEgMCAyLS45IDItMlY4bC02LTZ6bTIgMTZIOHYtMmg4djJ6bTAtNEg4di0yaDh2MnptLTMtNVYzLjVMMTguNSA5SDEzeiIvPjwvc3ZnPg==" alt="Project Report" />
  </a>
  <a href="https://www.rjwave.org/ijedr/viewpaperforall.php?paper=IJEDR2602755">
    <img src="https://img.shields.io/badge/Research_Paper-green?style=for-the-badge&logo=googledocs&logoColor=white" alt="Research Paper" />
  </a>
  <a href="https://github.com/yottajunaid/spiderhash/releases/download/v.1.0/spiderhash.exe">
    <img src="https://img.shields.io/badge/Download_EXE-0078D6?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48cGF0aCBmaWxsPSJ3aGl0ZSIgZD0iTTAgOTMuN2wxODMuNi0yNS4zdjE3Ny40SDBWOTMuN3ptMCAzMjQuNmwxODMuNiAyNS4zVjI2OC40SDB2MTQ5Ljl6bTIwMy44IDI4TDQ0OCA0ODBWMjY4LjRIMjAzLjh2MTc3Ljl6bTAtMzgwLjZ2MTgwLjFINDQ4VjMyTDIwMy44IDY1Ljd6Ii8+PC9zdmc+" alt="Windows EXE" />
  </a>
</div>

---

## Table of Contents

1. [Features](#features)  
2. [Supported Hash Algorithms](#supported-hash-algorithms)
3. [Installation](#installation) 
4. [How It Works (Architecture & Internals)](#how-it-works-architecture--internals)   
5. [Usage](#usage)
6. [Configuration](#configuration)  
7. [Performance & Testing](#performance--testing)  
8. [Limitations & Security Considerations](#limitations--security-considerations)  
9. [Contributing](#contributing)
10. [Project Report](#project-report)
11. [Research Paper](#research-paper)
12. [License](#license)  

---

## Features

- Graphical User Interface (GUI) to make hash cracking accessible without needing to write scripts.  
- Cracks many different types of hash algorithms (≈22).  
- Supports wordlist‐based cracking and possibly other modes (depending on algorithm).  
- Testing framework (there is a `TESTING_REPORT.xlsx`) showing which hashes has been tested just for reference. 

---

## Supported Hash Algorithms

| Algorithm      | Description                                                                 | Hash Length |
|----------------|-----------------------------------------------------------------------------|-------------|
| bcrypt         | Password-hashing function with salt and configurable cost. Widely used for secure password storage. | 60 chars (variable with cost) |
| MD5            | Legacy cryptographic hash (128-bit). Broken, but still seen in old systems. | 32 hex chars |
| SHA-1          | 160-bit cryptographic hash. Considered insecure today.                      | 40 hex chars |
| SHA-224        | 224-bit variant of SHA-2. Better than SHA-1, but less common.               | 56 hex chars |
| SHA-256        | Secure 256-bit SHA-2 variant, used in TLS and blockchain.                   | 64 hex chars |
| SHA-384        | Longer SHA-2 variant (384-bit).                                             | 96 hex chars |
| BLAKE2b        | Modern cryptographic hash optimized for 64-bit platforms.                   | 128 hex chars |
| BLAKE2s        | Lightweight version of BLAKE2 for smaller platforms.                        | 64 hex chars |
| SHA-512        | 512-bit SHA-2 variant. Strong, but slower than SHA-256.                     | 128 hex chars |
| SHA3-256       | Keccak-based SHA-3 (256-bit).                                               | 64 hex chars |
| SHA3-384       | Keccak SHA-3 variant (384-bit).                                             | 96 hex chars |
| SHA3-512       | Keccak SHA-3 variant (512-bit).                                             | 128 hex chars |
| RIPEMD-160     | 160-bit European hash function, historically used in Bitcoin addresses.     | 40 hex chars |
| SHAKE128-256   | SHA-3 extendable output function (XOF), truncated to 256 bits.              | 64 hex chars |
| SHAKE256-512   | SHA-3 XOF variant, truncated to 512 bits.                                   | 128 hex chars |
| CRC32 (HEX)    | Non-cryptographic checksum (32-bit). Used for error detection.              | 8 hex chars |
| Murmur3_32     | Fast non-cryptographic 32-bit hash for hash tables.                         | 8 hex chars |
| CityHash128    | Google’s fast non-cryptographic 128-bit hash.                               | 32 hex chars |
| XXH32          | Extremely fast 32-bit non-cryptographic hash (xxHash family).               | 8 hex chars |
| XXH64          | 64-bit variant of xxHash.                                                   | 16 hex chars |
| XXH3_64        | New xxHash3 variant, optimized for speed.                                   | 16 hex chars |
| XXH3_128       | 128-bit version of xxHash3.                                                 | 32 hex chars |

---

# Installation

## Windows

Download the `.exe` for windows:
https://github.com/yottajunaid/spiderhash/releases/tag/v.1.0

### Cross-Platform Compatibility (Wine)

While this executable is built natively for **Windows 8/10/11**, you can run it on **Linux** and **macOS** using **Wine**, a compatibility layer capable of running Windows applications.

To install Wine on your system, use the following commands:

#### For Linux (Ubuntu/Debian-based):

Bash

```
sudo apt update
sudo apt install wine
```

#### For macOS (via Homebrew):

Bash

```
brew install --cask wine-stable
```

**To run the application**: Navigate to the download folder in your terminal and execute: `wine SpiderHash.exe`

## Install from Source code

Supported Operating Systems
```bash
Windows 8/10/11
```

SpiderHash is designed specifically for Windows operating systems. Follow these steps to set it up:

**Clone the repository**
   ```bash
   git clone https://github.com/yottajunaid/spiderhash.git
   cd spiderhash
```
**Download and install Python 3.8 or later.**
You can easily install via Microsoft Store
<img width="1123" height="625" alt="Screenshot 2025-09-18 210724" src="https://github.com/user-attachments/assets/d6aba567-eae0-4671-b320-0bd126439e74" />


**Set up a virtual environment (recommended, but not necassary)**

```bash
python -m venv venv
venv\Scripts\activate
```
**Install dependencies**

Runtime dependencies:
```bash
pip install -r requirements.txt
```
Development dependencies:
```bash
pip install -r requirements_dev.txt
```
**Run SpiderHash**
```bash
python src\main.py
```

## How It Works (Architecture & Internals)

Here is a breakdown of how SpiderHash is structured under the hood, how it attempts to crack hashes, and how you might extend or modify it.

### Code Structure

- The repository has a `src/` directory where the main logic lives.  
- There are separate Python modules/files for:
  - GUI handling (windows, input, progress, result display)  
  - Hash algorithm handlers: each algorithm has code to verify / compute the hash given plaintext.  
  - Cracking engines: wordlist‐based, probabilistic, possibly brute‐force or hybrid.  
  - Utility modules: reading wordlists, validating format, progress reporting, maybe threading.  

### Crack Flow

1. **Input from user**: user supplies the hash (maybe multiple), chooses which algorithm they believe it to be, supplies wordlist or rules.  
2. **Pre-checks**: validate that hash format matches algorithm (length, hex/base64, etc.).  
3. **Attempt wordlist mode**: iterate over dictionary entries, hash them, compare with target.  
4. **Possibly brute force**: if code supports it, generate candidate strings (based on allowed character sets, lengths), hash, compare.  
5. **Result reporting**: show when a match is found; provide partial progress if running long; allow abort.  

### GUI

- Provides forms / dialogs for entering the target hash(es).  
- Drop-downs or controls to select algorithm.  
- File chooser to select wordlist.  
- Buttons to start / stop cracking.  
- Progress bars or log area to show status.  

### Dependencies

- Uses various Python libraries (to be found in `requirements.txt`). These might include:
  - `tkinter` or another GUI toolkit  
  - standard crypto/hash libraries (from Python’s `hashlib`)  
  - maybe third-party libs for some weaker or unusual hash types, or to speed up some computations.  
- For development, `requirements_dev.txt` probably includes testing tools (pytest, code linters, etc.).  

---

## Usage

### Running the GUI

After installation:

`python src/main.py`

Or if your entrypoint differs, specify accordingly. The GUI should open, allowing you to:

- Enter or paste one or more target hashes.
    
- Select the supposed algorithm.
    
- Provide a wordlist file (or select “brute-force / other” mode if needed).
    
- Run the cracking process; monitor status.

* * *

## Configuration

- Wordlists: specify path, maybe format (one password per line).
    
- Algorithm settings: maybe salt, iteration count (for KDFs), or format parsing.
    
- Character set / length bounds if brute force.
    
- GUI settings: logging.
    

* * *

## Performance & Testing

- SpiderHash includes a **testing report** (`TESTING_REPORT.xlsx`) which documents for various algorithms which hashes (from test set) get cracked with which wordlists, and which settings.
    
- For heavier hashes (SHA-512, etc), cracking may take large time depending on wordlist size or brute force space. SpiderHash performance depends heavily on:
    
    **1.  Hardware (CPU speed, number of cores)**
        
    **2.  Efficiency of hash implementation in Python / external libs**
        
    **3.  Size of wordlists / complexity of brute force**
        
- Suggestions: use smaller subsets or better probable wordlists, multi‐threading if implemented.
    

* * *

## Limitations & Security Considerations

Important to be clear (this is where things get philosophical + practical):

- Cracking hashes is *only* possible when preimage attacks are feasible: weak hashes, low entropy passwords, wordlist matches. For strong, salted, slow hashes, or high entropy passwords, cracking may be infeasible.
    
- GUI apps are easier to misuse or leak secrets; handle sensitive data with care.
    
- Legal / ethical constraints: only crack hashes you are authorized to.
    
- Performance may be very low for bruteforce of large keyspaces (e.g. long passwords, mixed chars).
    

* * *

## Contributing

If you’d like to contribute:

- Fork the repo.
    
- Work on a feature branch.
    
- Update or add tests.
    
- Ensure backwards compatibility with existing algorithms.
    
- Submit pull requests.

* * *

## Project Report
Download: [spiderhash_report.pdf](https://github.com/user-attachments/files/27314192/spiderhash_report.pdf)

* * *
## Research Paper

Download: [Paper](https://www.rjwave.org/ijedr/viewpaperforall.php?paper=IJEDR2602755)

## License

This project is licensed under **GPL-3.0**. (See LICENSE file)

* * *

## Acknowledgments

- Thanks to all open source libraries (Python's `hashlib`, GUI toolkit, etc.)
    
- Thanks to testers who contributed the reports and benchmarks.

- Thanks to the Open-Source Developers who build the Logic Code, Mathematics, workflow, etc
    

* * *

## Example

Here’s a typical example session:

1.  User launches the GUI.
    
2.  Enters hash: `5f4dcc3b5aa765d61d8327deb882cf99`
    
3.  Chooses algorithm: **MD5**
    
4.  Loads wordlist: `common_passwords.txt`
    
5.  Clicks **Start** → status bar shows progress, maybe attempts per second, etc.
    
6.  Cracker finds match: `password` → displays to user.
    

* * *

### Contact / Support

- Report bugs or request features via GitHub Issues.
    
- For questions / help, can email / reach out to **yottajunaid** [(or via portfolio)](https://yottajunaid.github.io/portfolio/).
