# Cybersecurity Dataset Data Dictionary
## July 2011 - December 2024

---

## Dataset Overview

This comprehensive cybersecurity dataset aggregates monthly temporal data from multiple authoritative sources spanning July 2011 to December 2024. The dataset integrates attack incidents, defence solutions, geopolitical conflict indicators, and underground market intelligence across 37 countries.

### Temporal Coverage
- **Date Range**: July 2011 - December 2024 (monthly granularity)
- **Frequency**: Monthly observations
- **Total Countries Tracked**: 37 nations plus global aggregates

### Data Sources
1. **Hackmageddon** - Cyber attack incident tracking (2011-2024)
2. **Elsevier** - Academic literature mentions of attacks and Pertinent Alleviation Technologies (PATs) (2011-2024)
3. **University of Maryland Cyber Event Database** - Attack type classifications (2014-2024)
4. **National Vulnerability Database (NVD/CVE)** - Vulnerability metadata and attack descriptions (2011-2024)
5. **Dark Web Marketplaces & Forums** - Underground market data (2020-2024)
6. **RSS Feeds** - 138 cybersecurity news feeds for attacks and solutions (2011-2024)
7. **Google Trends** - Global search interest for attacks and solutions (2011-2024)
8. **Twitter** - War/conflict mentions (2011-2022)
9. **Bluesky** - War/conflict mentions (2023-2024)
10. **ACLED** - Political violence event tracking (2017-2024)
11. **Python Holidays** - National holiday indicators (2011-2024)

---

## Country Codes

The following ISO 3166-1 alpha-2 country codes are used throughout the dataset:

| Code | Country | Code | Country | Code | Country |
|------|---------|------|---------|------|---------|
| US | United States | GB | United Kingdom | CA | Canada |
| AU | Australia | UA | Ukraine | RU | Russia |
| FR | France | DE | Germany | BR | Brazil |
| CN | China | JP | Japan | PK | Pakistan |
| KP | North Korea | KR | South Korea | IN | India |
| TW | Taiwan | NL | Netherlands | ES | Spain |
| SE | Sweden | MX | Mexico | IR | Iran |
| IL | Israel | SA | Saudi Arabia | SY | Syria |
| FI | Finland | IE | Ireland | AT | Austria |
| NO | Norway | CH | Switzerland | IT | Italy |
| MY | Malaysia | EG | Egypt | TR | Turkey |
| PT | Portugal | PS | Palestine | AE | United Arab Emirates |
| ? | Unknown/Unattributed | ALL | Global Total |

---

## Data Field Categories

### 1. Hackmageddon Attack Type Data
**Naming Convention**: `{Attack Type}-{Country Code}`

**Description**: Monthly count of cybersecurity attack incidents by type and target country, aggregated from Hackmageddon's incident tracking database.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Attack Types**:

#### 1.1 Network-Based Attacks
- **DDoS** - Distributed Denial of Service attacks overwhelming network resources
- **Phishing** - Social engineering attacks via fraudulent communications
- **SQL Injection** - Database manipulation through malicious SQL queries
- **XSS** - Cross-Site Scripting attacks injecting malicious scripts into web pages
- **Password Attack** - Credential compromise attempts including credential stuffing and dictionary attacks

#### 1.2 Malware Attacks
- **Ransomware** - Malicious software encrypting data for ransom
- **Trojan** - Malware disguised as legitimate software
- **Malware** - General malicious software category
- **Adware** - Unwanted advertising software
- **Worms** - Self-replicating malware spreading across networks
- **Spyware** - Software covertly collecting user information
- **Backdoor** - Unauthorised remote access mechanisms
- **Botnet** - Networks of compromised devices controlled remotely
- **Cryptojacking** - Unauthorised cryptocurrency mining on victim systems

#### 1.3 Advanced Attacks
- **Advanced persistent threat** - Prolonged targeted attacks by sophisticated actors
- **Zero-day** - Exploits of previously unknown vulnerabilities
- **Targeted Attack** - Attacks against specific entities or individuals
- **Account Hijacking** - Unauthorised takeover of user accounts

#### 1.4 Web & Application Attacks
- **Defacement** - Unauthorised alteration of website appearance
- **Vulnerability** - Exploitation of system weaknesses
- **Brute Force Attack** - Automated credential guessing attacks
- **Malvertising** - Malicious advertising injecting malware

#### 1.5 Information Operations
- **Data Breach** - Unauthorised access and exfiltration of sensitive data
- **Disinformation/Misinformation** - False or misleading information campaigns

#### 1.6 Other Categories
- **Unknown** - Attacks with unidentified characteristics
- **Others** - Attacks not fitting standard categories

**Example Fields**:
- `DDoS-US`: Count of DDoS attacks targeting United States infrastructure
- `Ransomware-GB`: Count of ransomware incidents in the United Kingdom
- `Phishing-ALL`: Global total of phishing attacks across all countries

---

### 2. Elsevier Attack Mentions Data
**Naming Convention**: `Mentions-{Attack Type}`

**Description**: Monthly count of attack type mentions in academic cybersecurity literature indexed by Elsevier. Represents research attention and discourse around specific attack vectors.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Attack Types**: Includes all Hackmageddon categories plus specialised threats:

#### 2.1 Additional Specialised Attacks
- **MITM** - Man-in-the-Middle attacks intercepting communications
- **DNS Spoofing** - Domain Name System cache poisoning
- **Pegasus Spyware** - Sophisticated mobile spyware variant
- **CoolWebSearch Spyware** - Browser hijacking spyware
- **Gator GAIN Spyware** - Adware and spyware variant
- **180search Assistant Spyware** - Browser toolbar spyware
- **Transponder vx2 Spyware** - Browser hijacking spyware
- **WannaCry Ransomware** - Notable 2017 global ransomware outbreak
- **Colonial Pipeline Ransomware** - 2021 critical infrastructure attack
- **Cryptolocker** - Ransomware variant encrypting files
- **Dropper** - Malware installing other malicious payloads
- **Wiper** - Malware permanently destroying data
- **Pharming** - Redirecting users to fraudulent websites
- **Insider Threat** - Attacks by authorised users
- **Drive-by** - Malware infections via compromised websites
- **Rootkit** - Malware concealing its presence in systems
- **Adversarial Attack** - Attacks against machine learning models
- **Data Poisoning** - Corrupting training data for ML systems
- **Deepfake** - AI-generated synthetic media for deception
- **Deeplocker** - AI-powered targeted malware
- **Supply Chain** - Attacks via compromised vendors or software
- **IoT Device Attack** - Exploits targeting Internet of Things devices
- **Keylogger** - Software recording keyboard inputs
- **DNS Tunneling** - Covert data exfiltration via DNS protocols
- **Session Hijacking** - Unauthorised takeover of active sessions
- **URL manipulation** - Exploiting URL parameters for attacks

**Example Fields**:
- `Mentions-DDoS`: Academic paper mentions of DDoS attacks
- `Mentions-Ransomware`: Research references to ransomware
- `Mentions-Supply Chain`: Literature discussing supply chain attacks

---

### 3. War/Conflict Data
**Naming Convention**: `WAR/CONFLICT {Country Code}`

**Description**: Monthly count of war or conflict-related mentions associated with each country, aggregated from Twitter (2011-2022) and Bluesky (2023-2024) social media posts. Indicates geopolitical tensions potentially correlating with cyber activity.

**Data Type**: Integer (count)

**Temporal Coverage**: 
- Twitter: July 2011 - December 2022
- Bluesky: January 2023 - December 2024

**Example Fields**:
- `WAR/CONFLICT US`: Monthly mentions of US-related conflicts
- `WAR/CONFLICT UA`: Monthly mentions of Ukraine-related conflicts
- `WAR/CONFLICT ALL`: Global total of all conflict mentions

---

### 4. Holiday Indicators
**Naming Convention**: `{Country Code}_holiday`

**Description**: Monthly count of holidays in a given month, contains national public holidays for the specified country. Generated using the Python Holidays library.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Additional Field**:
- `Holidays`: Aggregate count of holidays across all tracked countries for the month

**Example Fields**:
- `US_holiday`: Indicator for US federal holidays
- `GB_holiday`: Indicator for UK bank holidays
- `Holidays`: Total number of holidays globally that month

---

### 5. Elsevier Pertinent Alleviation Technologies (PATs) Mentions
**Naming Convention**: `Solution_{Technology Name}_Mentions`

**Description**: Monthly count of cybersecurity defence technology mentions in Elsevier-indexed academic literature. Represents research focus on defensive measures and security controls.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Technology Categories**:

#### 5.1 Cryptographic Technologies
- **BLOCKCHAIN** - Distributed ledger technology
- **ENCRYPTION** - Data confidentiality measures
- **CRYPTOGRAPHY** - Mathematical security foundations
- **SSL/TLS** - Secure Sockets Layer/Transport Layer Security
- **HTTPS** - HTTP Secure protocol
- **Identity-Based Encryption (IBE)** - Public key encryption variant
- **MERKLE SIGNATURE** - Post-quantum signature schemes
- **DNSSEC** - DNS Security Extensions
- **CERTIFICATE PINNING** - PKI security enhancement

#### 5.2 Access Control & Authentication
- **ACCESS CONTROL** - Permission management systems
- **IDENTITY MANAGEMENT** - User identity lifecycle management
- **MULTI FACTOR AUTHENTICATION** - Multiple credential verification
- **LEAST PRIVILEGE** - Minimal necessary access principle
- **STRONG AUTHENTICATION** - Robust credential verification
- **CONTINUOUS AUTHENTICATION** - Ongoing user verification
- **MUTUAL AUTHENTICATION** - Bidirectional identity verification
- **ONE TIME PASSWORD** - Single-use authentication codes
- **BIOMETRICS** - Biological characteristic authentication
- **KEYSTROKE DYNAMICS** - Typing pattern authentication
- **GRAPHICAL AUTHENTICATION** - Image-based login systems
- **PASSWORD HASHING** - Secure password storage
- **PASSWORD STRENGTH METERS** - Password quality indicators
- **PASSWORD MANAGEMENT** - Credential storage solutions
- **PASSWORD POLICY** - Password requirements enforcement

#### 5.3 Threat Detection & Analysis
- **ML/DL** - Machine Learning/Deep Learning for security
- **ANOMALY DETECTION** - Unusual behaviour identification
- **IDS/IPS** - Intrusion Detection/Prevention Systems
- **STATIC ANALYSIS** - Code analysis without execution
- **DYNAMIC ANALYSIS** - Runtime behaviour analysis
- **BEHAVIOR BASED DETECTION** - Profiling normal vs malicious activity
- **OUTLIER DETECTION** - Statistical anomaly identification
- **USER BEHAVIOR ANALYTICS** - Analysing user activity patterns
- **FILE INTEGRITY MONITORING** - Detecting unauthorised file changes
- **SIEM** - Security Information and Event Management
- **ACTIVITY MONITORING** - Tracking system and user actions
- **DARKNET MONITORING** - Surveillance of underground forums
- **VULNERABILITY SCANNER** - Automated security assessment tools
- **TAINT ANALYSIS** - Tracking untrusted data flow

#### 5.4 Network Security Technologies
- **HONEYPOT** - Decoy systems for attacker analysis
- **SOFTWARE DEFINED NETWORK** - Programmable network infrastructure
- **TRAFFIC SHAPING** - Network bandwidth management
- **PACKET FILTERING** - Network traffic inspection
- **BLACKHOLING** - Routing malicious traffic to null routes
- **NETWORK SEGMENTATION** - Dividing networks for isolation
- **VPN** - Virtual Private Networks
- **STANDARDIZED COMMUNICATION** - Secure protocol implementations

#### 5.5 Application Security
- **PENETRATION TESTING** - Authorised attack simulations
- **VULNERABILITY MANAGEMENT** - Weakness identification and remediation
- **VULNERABILITY ASSESSMENT** - Security posture evaluation
- **DATA SANITIZATION** - Input validation and cleaning
- **SESSION MANAGEMENT** - Secure session handling
- **CAPTCHA** - Challenge-response tests for bots
- **BLACKLISTING** - Blocking known malicious entities
- **RATE LIMITING** - Request throttling mechanisms
- **SANDBOXING** - Isolated execution environments
- **CODE SIGNING** - Software authenticity verification
- **APPLICATION WHITELISTING** - Allowing only approved software
- **SECURE BOOT** - Verified system startup
- **PATCH MANAGEMENT** - Software update deployment
- **Control Flow Integrity** - Preventing code execution hijacks

#### 5.6 Data Protection
- **SUPPLY CHAIN RISK MANAGEMENT** - Vendor security oversight
- **DATA PROVENANCE** - Tracking data origins and transformations
- **PRIVACY PRESERVING** - Privacy-enhancing technologies
- **DATA BACKUPS** - Redundant data storage
- **DATA LOSS PREVENTION** - Preventing unauthorised data exfiltration
- **DATA LEAKAGE DETECTION/PREVENTION** - Monitoring for data exposure
- **DIGITAL WATERMARK** - Embedded ownership markers

#### 5.7 Advanced & Emerging Technologies
- **GAME THEORY** - Strategic security decision modelling
- **GRAPHICAL MODEL** - Probabilistic relationship modelling
- **RANK CORRELATION** - Statistical dependency measures
- **Bayesian Network** - Probabilistic graphical models
- **FORMAL VERIFICATION** - Mathematical correctness proofs
- **ADVERSARIAL TRAINING** - ML model hardening
- **TRUSTWORTHY AI** - Reliable and ethical AI systems
- **HIDDEN MARKOV MODEL** - Sequential data modelling
- **DIMENSIONALITY REDUCTION** - Feature space compression
- **DEFENSIVE DISTILLATION** - Neural network defence technique
- **RRAM** - Resistive RAM for hardware security
- **SPATIAL SMOOTHING** - Image processing defence
- **NOISE INJECTION** - Adding random data for privacy
- **RISK ASSESSMENT** - Threat likelihood and impact analysis
- **MOVING TARGET DEFENSE** - Dynamic system configuration
- **DECEPTION TECHNOLOGY** - Active adversary misdirection
- **ATTACK TREE** - Hierarchical threat modelling
- **HYPERGAME** - Game theory with information asymmetry
- **NLP/LLM** - Natural Language Processing/Large Language Models for security

#### 5.8 IoT & Hardware Security
- **SPLIT MANUFACTURING** - Hardware supply chain security
- **LIVENESS DETECTION** - Verifying physical user presence
- **3D FACE RECONSTRUCTION** - Biometric anti-spoofing
- **PUBLIC KEY INFRASTRUCTURE** - Certificate management systems
- **SECURE SIMPLE PAIRING** - Bluetooth security protocol
- **DYNAMIC BINARY INSTRUMENTATION** - Runtime code analysis
- **DISTRIBUTED LEDGERS** - Decentralised record systems
- **SOURCE IDENTIFICATION** - Tracing data origins
- **IMAGE RECOGNITION** - Visual authentication and analysis

**Example Fields**:
- `Solution_ENCRYPTION_Mentions`: Academic mentions of encryption technologies
- `Solution_ML/DL_Mentions`: References to machine learning security applications
- `Solution_BLOCKCHAIN_Mentions`: Blockchain security research mentions

---

### 6. ACLED Political Violence Events
**Naming Convention**: `{Country Code}_political_violence_events`

**Description**: Monthly count of political violence incidents from the Armed Conflict Location & Event Data Project (ACLED), including protests, riots, battles, and violence against civilians.

**Data Type**: Integer (count)

**Temporal Coverage**: January 2017 - December 2024

**Additional Field**:
- `Political Violence Incidents (ALL)`: Aggregate count across all countries

**Example Fields**:
- `US_political_violence_events`: Political violence incidents in the United States
- `SY_political_violence_events`: Political violence incidents in Syria

---

### 7. Google Trends Attack Data
**Naming Convention**: `{Attack Term}: (Attacks_Google_Trends_Worldwide)`

**Description**: Monthly normalised search interest (0-100 scale) for cybersecurity attack terms on Google Search worldwide. Values represent search volume relative to the peak search interest within the dataset timeframe.

**Data Type**: Integer (0-100 scale)

**Temporal Coverage**: July 2011 - December 2024

**Attack Terms Tracked**:
- Denial-of-service attack
- Phishing
- Ransomware
- SQL injection
- Website defacement
- Trojan horse
- Vulnerability
- Zero-day vulnerability
- Advanced persistent threat
- Cross-site scripting
- Malware
- Data breach
- Adware
- Brute-force attack
- Malvertising
- Backdoor
- Botnet
- Cryptojacking
- Worm drive
- Spyware (appears twice in source)
- Computer virus
- Man-in-the-middle attack
- Insider threat
- Pharming
- adversarial attack
- Rootkit
- Session hijacking
- CryptoLocker

**Example Fields**:
- `Denial-of-service attack: (Attacks_Google_Trends_Worldwide)`: Global DDoS search interest
- `Ransomware: (Attacks_Google_Trends_Worldwide)`: Global ransomware search interest

---

### 8. Google Trends Solutions Data
**Naming Convention**: `{Solution Term}: (Solutions_Google_Trends_Worldwide)`

**Description**: Monthly normalised search interest (0-100 scale) for cybersecurity defence technologies on Google Search worldwide.

**Data Type**: Integer (0-100 scale)

**Temporal Coverage**: July 2011 - December 2024

**Solution Terms Tracked**:
- Anomaly detection
- Cryptography
- Penetration Testing Execution Standard
- Host-based intrusion detection system
- Multi-factor authentication
- Principle of least privilege
- CAPTCHA
- Rate limiting
- Graphical model
- Honeypot
- Software-defined networking
- Traffic shaping
- Black Holing
- Rank correlation
- Kerberos
- Secure Sockets Layer
- Identity-based encryption
- Data sanitization
- AI Security Measures
- Trustworthy AI
- Bayesian network
- Formal verification
- Vulnerability management
- Virtual private network
- Secure Boot
- Merkle signature scheme
- biometrics
- Digital watermarking
- Hidden Markov model
- Patch management
- Dimensionality reduction
- Noise Injection
- Network segmentation
- User behavior analytics
- Deception technology
- Virtual keyboard
- Code signing
- File signature
- Public key infrastructure
- Mutual authentication
- One-time password
- Domain Name System Security Extensions
- Functional Encryption
- Vulnerability assessment
- Security information and event management
- Control-flow integrity
- Vulnerability scanner
- Password policy
- Data Loss Prevention
- Moving Target Defense
- Keystroke dynamics
- Attack tree
- Distributed ledger
- Large language model
- Natural language processing


**Example Fields**:
- `Anomaly detection: (Solutions_Google_Trends_Worldwide)`: Global anomaly detection search interest
- `Multi-factor authentication: (Solutions_Google_Trends_Worldwide)`: Global MFA search interest

---

### 9. University of Maryland Cyber Event Database
**Naming Convention**: `{Event Type}-{Country Code}`

**Description**: Monthly count of cyber events classified by the University of Maryland's Cyber Event Database according to their technical attack methodology.

**Data Type**: Integer (count)

**Temporal Coverage**: January 2014 - December 2024

**Event Type Classifications**:

#### 9.1 Infrastructure Attacks
- **Exploitation of Application Server** - Compromising application-layer servers
- **Exploitation of End Hosts** - Targeting user devices and workstations
- **Exploitation of Network Infrastructure** - Attacking routers, switches, and core network components
- **Exploitation of Network Server** - Compromising network-layer servers
- **Exploitation of Sensors** - Targeting IoT and monitoring devices

#### 9.2 Service Disruption
- **External Denial of Service** - DDoS attacks from outside networks
- **Internal Denial of Service** - DoS attacks originating within networks

#### 9.3 Data & Communication Attacks
- **Data Attack** - Direct targeting of data integrity or availability
- **Message Manipulation** - Altering communications in transit
- **Exploitation of Data in Transit** - Intercepting network traffic

#### 9.4 Physical & Miscellaneous
- **Physical Attack** - Hardware tampering or physical security breaches
- **Undetermined** - Events with unclear attack methodology

**Example Fields**:
- `Exploitation of Application Server-US`: Application server attacks in the United States
- `External Denial of Service-CN`: External DDoS attacks against China
- `Data Attack-ALL`: Global total of data-targeted attacks

---

### 10. Dark Web Marketplace Monthly Counts
**Naming Convention**: `{Attack/Tool}_DARK_WEB_MARKETPLACE_MONTHLY COUNTS`

**Description**: Monthly count of mentions, listings, or discussions of specific attack types and tools on dark web marketplaces and associated forums. Indicates underground market trends and threat actor interest.

**Data Type**: Integer (count)

**Temporal Coverage**: January 2020 - December 2024

**Tracked Attack Types & Tools**:

#### 10.1 Malware & Exploits
- **Malware** - General malicious software mentions
- **Ransomware** - Ransomware variants and RaaS offerings
- **Trojan** - Trojan horse malware
- **Spyware** - Surveillance software
- **Virus** - Self-replicating malware
- **Worm** - Network-spreading malware
- **Adware** - Advertising malware
- **Botnet** - Compromised device networks
- **RAT** - Remote Access Trojans
- **Keylogger** - Keystroke recording tools
- **Dropper** - Malware installation tools
- **Downloader** - Payload retrieval tools

#### 10.2 Specific Malware Families
- **Redline Stealer** - Information-stealing malware
- **Cryptolocker** - Ransomware variant
- **Mirai** - IoT botnet malware
- **Conti** - Ransomware-as-a-Service operation
- **REvil** - Ransomware gang and malware
- **DarkSide** - Ransomware gang (Colonial Pipeline attackers)
- **BlackCat** - ALPHV ransomware variant
- **Maze** - Data-leak ransomware variant
- **Pegasus** - Commercial spyware
- **Predator** - Commercial spyware

#### 10.3 Attack Methodologies
- **DDoS** - Distributed Denial of Service
- **Phishing** - Social engineering attacks
- **SQL Injection** - Database attacks
- **XSS** - Cross-site scripting
- **Brute Force** - Credential guessing
- **Dictionary Attack** - Password attack using wordlists
- **Zero-day** - Unknown vulnerability exploits
- **Doxing** - Personal information exposure
- **Social Engineering** - Psychological manipulation

#### 10.4 Infrastructure & Tools
- **Shell** - Web shells for remote access
- **Backdoor** - Persistent access mechanisms
- **Bootkit** - Boot-level rootkits
- **C2 Server** - Command and Control infrastructure
- **Exploit Kit** - Automated exploitation frameworks
- **EK** - Exploit Kit (abbreviated)

#### 10.5 Exploit Kit Families
- **Angler** - Exploit kit variant
- **Rig** - Exploit kit variant
- **Neutrino** - Exploit kit variant

#### 10.6 Advanced Threats
- **Advanced Persistent Threat** - Sophisticated prolonged attacks
- **Remote Code Execution** - RCE vulnerabilities
- **Privilege Escalation** - Elevation of access rights
- **Persistence** - Maintaining long-term access

#### 10.7 Financial Crime
- **Carding** - Credit card fraud
- **Card Fraud** - Payment card compromise
- **BIN** - Bank Identification Number data
- **CVV** - Card verification value codes
- **Fullz** - Complete identity information packages
- **Bank Log** - Banking credential sets
- **Dump** - Stolen card magnetic stripe data

#### 10.8 Account Compromise
- **ATO** - Account Takeover
- **Credential Stuffing** - Automated login attempts
- **Cookie Stealing** - Session token theft

#### 10.9 Vulnerabilities & Attacks
- **Vulnerability** - Software weaknesses
- **RFI** - Remote File Inclusion
- **LFI** - Local File Inclusion
- **SQL** - SQL-related attacks
- **Data Breach** - Unauthorised data access
- **Data Leak** - Exposed sensitive information
- **Leaked Database** - Compromised database dumps

#### 10.10 Underground Economy
- **Dark Web Market** - Marketplace platforms
- **Hacking Forum** - Discussion platforms
- **Carding Forum** - Card fraud communities
- **Insider Fraud** - Internal threat activity
- **BEC** - Business Email Compromise
- **Email Spoofing** - Forged sender addresses
- **Email Flood** - Email bombing attacks

**Example Fields**:
- `Ransomware_DARK_WEB_MARKETPLACE_MONTHLY COUNTS`: Ransomware mentions on dark web
- `DDoS_DARK_WEB_MARKETPLACE_MONTHLY COUNTS`: DDoS service offerings
- `CVV_DARK_WEB_MARKETPLACE_MONTHLY COUNTS`: Stolen card data listings

---

### 11. Dark Web Marketplace Pricing Data
**Naming Convention**: `{Attack/Tool}_DARK_WEB_MARKETPLACE_MARKET_PRICE`

**Description**: Average monthly price (in USD) for attack tools, services, or compromised data on dark web marketplaces. Reflects underground market economics and demand.

**Data Type**: Float (USD)

**Temporal Coverage**: January 2020 - December 2024

**Priced Items**: Same categories as Dark Web Monthly Counts (Section 10)

**Note**: Pricing may vary significantly based on:
- Target country or organisation
- Data quality or tool sophistication
- Market competition and supply
- Cryptocurrency exchange rates
- Law enforcement actions affecting supply

**Example Fields**:
- `Ransomware_DARK_WEB_MARKETPLACE_MARKET_PRICE`: Average RaaS subscription cost
- `CVV_DARK_WEB_MARKETPLACE_MARKET_PRICE`: Average price per stolen card number
- `Zero-day_DARK_WEB_MARKETPLACE_MARKET_PRICE`: Average zero-day exploit price

---

### 12. RSS Feed Attack Mentions
**Naming Convention**: `{Attack Type}_RSS_FEEDS`

**Description**: Monthly count of attack type mentions across 138 monitored cybersecurity RSS news feeds. Indicates media attention and incident reporting.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Attack Types Tracked**:
- DDoS
- Phishing
- Ransomware
- Vulnerability
- Zero-day
- APT (Advanced Persistent Threat)
- Malware
- Data_Breach
- Account_Hijacking
- Defacement
- Spyware
- Unknown
- Others
- Trojan
- Password_Attack
- SQL_Injection
- XSS
- Disinformation
- Targeted_Attack
- Adware
- Brute_Force
- Malvertising
- Backdoor
- Botnet
- Cryptojacking
- Worm
- MITM
- DNS_Spoofing
- Insider_Threat
- Drive-by
- Rootkit
- Adversarial_Attack
- Data_Poisoning
- Deepfake
- Deeplocker
- Supply_Chain
- IoT_Attack
- DNS_Tunneling
- Session_Hijacking
- URL_Attack
- Unknown_Attack

**Special Field**:
- `ALL_VULNERABILITIES_RSS_FEEDS`: Aggregate count of all vulnerability mentions

**Example Fields**:
- `Ransomware_RSS_FEEDS`: Ransomware mentions in cybersecurity news
- `Data_Breach_RSS_FEEDS`: Data breach incident reports
- `Supply_Chain_RSS_FEEDS`: Supply chain attack coverage

---

### 13. RSS Feed Solution Mentions
**Naming Convention**: `{Solution Technology}_RSS_FEEDS`

**Description**: Monthly count of cybersecurity defence technology mentions across 138 monitored RSS feeds. Indicates industry discussion and adoption trends.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Solution Technologies Tracked**:

#### 13.1 Core Security Technologies
- BLOCKCHAIN
- ACCESS CONTROL
- ENCRYPTION
- SUPPLY CHAIN RISK MANAGEMENT
- IDENTITY MANAGEMENT
- MACHINE LEARNING
- ANOMALY DETECTION
- CRYPTOGRAPHY
- PENETRATION TESTING
- INTRUSION DETECTION PREVENTION
- STATIC ANALYSIS
- DYNAMIC ANALYSIS

#### 13.2 Authentication & Access
- MULTI FACTOR AUTHENTICATION
- LEAST PRIVILEGE
- SESSION MANAGEMENT
- SESSION ID RANDOMIZATION
- STRONG AUTHENTICATION
- CONTINUOUS AUTHENTICATION
- MUTUAL AUTHENTICATION
- PASSWORD SALT
- ONE TIME PASSWORD
- VOICEPRINT AUTHENTICATION
- PASSWORD HASH
- PASSWORD STRENGTH METER
- PASSWORD MANAGEMENT
- PASSWORD POLICY
- GRAPHICAL AUTHENTICATION

#### 13.3 Network Security
- CAPTCHA
- BLACKLISTING
- RATE LIMITING
- HONEYPOT
- SOFTWARE DEFINED NETWORK
- TRAFFIC SHAPING
- PACKET FILTERING
- BLACKHOLING
- D3NS DNS
- VPN
- NETWORK SEGMENTATION
- IP WHITELIST

#### 13.4 Advanced Analytics
- GRAPHICAL MODEL
- GAME THEORY
- GRAPH MACHINE LEARNING
- RANK CORRELATION
- OUTLIER DETECTION
- Bayesian Network
- DEEP PROBABILISTIC MODEL
- HIDDEN MARKOV MODEL
- BEHAVIOR BASED DETECTION

#### 13.5 Data Protection
- DATA SANITIZATION
- DATA PROVENANCE
- DATA BACKUP
- DATA LOSS PREVENTION
- DATA LEAKAGE DETECTION
- PRIVACY PRESERVING
- DATA AUGMENTATION

#### 13.6 Application Security
- SECURE SOCKETS LAYER
- HTTPS
- Identity-Based Encryption
- VULNERABILITY MANAGEMENT
- FILE INTEGRITY MONITORING
- CSS MATCHING
- URI MATCHING
- SECURE BOOT
- MERKLE SIGNATURE
- CODE SIGN
- FILE SIGNATURE
- PUBLIC KEY INFRASTRUCTURE
- DNSSEC
- CERTIFICATE PINNING
- SECURE SIMPLE PAIRING
- VULNERABILITY ASSESSMENT
- VULNERABILITY SCAN
- APPLICATION WHITELISTING
- PATCH MANAGEMENT
- Control Flow Integrity

#### 13.7 AI & ML Security
- ADVERSARIAL TRAINING
- TRUSTWORTHY AI
- LIVENESS DETECTION
- AUDIO DEEPFAKE DETECTION
- 3D FACE RECONSTRUCTION
- DIMENSIONALITY REDUCTION
- DEFENSIVE DISTILLATION
- GRADIENT MASKING
- SPATIAL SMOOTHING
- NOISE INJECTION
- IMAGE RECOGNITION
- NATURAL LANGUAGE PROCESSING
- LLM
- MAUVE
- PREBUNKING

#### 13.8 Hardware & IoT Security
- BIOMETRICS
- DIGITAL WATERMARK
- TROJAN ISOLATION
- HARDWARE SANDBOXING
- FORMAL VERIFICATION
- SPLIT MANUFACTURING
- RRAM
- Double Patterning Lithography

#### 13.9 Threat Intelligence & Response
- SANDBOXING
- DARKNET MONITORING
- USER BEHAVIOR ANALYTICS
- DECEPTION TECHNOLOGY
- RISK ASSESSMENT
- LOG CORRELATION
- DYNAMIC RESOURCE MANAGEMENT
- SIEM
- ACTIVITY MONITORING
- MOVING TARGET DEFENSE
- ATTACK TREE
- Automatic Violation Prevention

#### 13.10 Specialised Technologies
- TAINT ANALYSIS
- DYNAMIC BINARY INSTRUMENTATION
- ORTHOGONAL OBFUSCATION
- STANDARDIZED COMMUNICATION
- DISTRIBUTED LEDGER
- SOURCE IDENTIFICATION
- VIRTUAL KEYBOARD
- KEYSTROKE DYNAMICS
- HYPERGAME

**Special Field**:
- `ALL_SOLUTIONS_RSS_FEEDS`: Aggregate count of all solution technology mentions

**Example Fields**:
- `ENCRYPTION_RSS_FEEDS`: Encryption technology mentions in news
- `MULTI FACTOR AUTHENTICATION_RSS_FEEDS`: MFA coverage in cybersecurity media
- `MACHINE LEARNING_RSS_FEEDS`: ML security applications in news

---

### 14. National Vulnerability Database (NVD) Mentions
**Naming Convention**: `{Attack Category}_NVD_MENTIONS`

**Description**: Monthly count of attack type or threat category mentions in CVE (Common Vulnerabilities and Exposures) descriptions and associated NVD metadata. Represents formal vulnerability disclosures and their associated threat contexts.

**Data Type**: Integer (count)

**Temporal Coverage**: July 2011 - December 2024

**Attack Categories in NVD Data**:

#### 14.1 Social Engineering & Initial Access
- **Phishing** - Credential theft via deceptive communications
- **Social Engineering** - Psychological manipulation tactics
- **Email-Based Attacks** - Email as attack vector
- **Drive-by & Watering Hole** - Compromised website attacks

#### 14.2 Exploitation Vectors
- **Exploit Kits** - Automated exploitation toolkits
- **Zero-day** - Previously unknown vulnerabilities
- **Web App Attacks** - Application-layer exploits
- **Code Execution** - Arbitrary code execution vulnerabilities
- **DNS Attacks** - Domain Name System exploits
- **Password Attacks** - Credential compromise methods

#### 14.3 Malware Categories
- **Malware** - General malicious software
- **Ransomware** - Data encryption extortion
- **RaaS** - Ransomware-as-a-Service models
- **Ransomware Variants** - Specific ransomware families
- **Trojans** - Deceptive malicious programs
- **Backdoors** - Persistent unauthorised access
- **Info-Stealers** - Information exfiltration malware
- **Spyware Variants** - Surveillance software types
- **Worms & Viruses** - Self-propagating malware
- **Botnets** - Compromised device networks
- **Rootkits** - System-level concealment malware
- **Adware** - Advertising-focused malware
- **Wiper Malware** - Data destruction tools
- **MaaS** - Malware-as-a-Service offerings

#### 14.4 Network Attacks
- **DoS** - Denial of Service attacks
- **MITM** - Man-in-the-Middle interception

#### 14.5 Financial Crime
- **Carding** - Credit card fraud operations
- **Account Takeover** - Unauthorised account access
- **Cryptojacking** - Unauthorised cryptocurrency mining

#### 14.6 Data Compromise
- **Data Breaches** - Unauthorised data access events
- **Underground Markets** - Dark web trading platforms
- **Doxing** - Personal information exposure

#### 14.7 Advanced Threats
- **Lateral Movement** - Post-compromise network traversal
- **APTs** - Advanced Persistent Threats
- **Supply Chain** - Third-party compromise attacks
- **Insider Threats** - Internal malicious actors

#### 14.8 Emerging Threats
- **IoT Attacks** - Internet of Things exploits
- **AI Security** - Machine learning threats

**Special Field**:
- `Total_NVD_MENTIONS`: Sum of all attack category mentions in NVD records for the month

**Example Fields**:
- `Ransomware_NVD_MENTIONS`: Ransomware references in CVE descriptions
- `Zero-day_NVD_MENTIONS`: Zero-day vulnerabilities disclosed
- `Supply Chain_NVD_MENTIONS`: Supply chain vulnerabilities documented

---

## Data Quality & Usage Notes

### Missing Values
- Fields may contain zeros where no incidents/mentions occurred
- Pre-2014 University of Maryland data will be absent (source begins 2014)
- Pre-2017 ACLED data will be absent (source begins 2017)
- Pre-2020 Dark Web data will be absent (source begins 2020)
- Pre-2023 Bluesky data will be absent (platform launched 2023)
- Post-2022 Twitter data will be absent (source ends 2022)

### Country Code Interpretation
- **"?"** indicates attacks/events with unknown geographic attribution
- **"ALL"** represents global totals across all countries
- Some countries may have sparse data depending on attack types and sources

### Data Aggregation
- All temporal data is aggregated at monthly granularity
- Counts represent incidents, mentions, or events within the calendar month
- Google Trends data is normalised (0-100 scale) relative to peak interest
- Dark Web pricing is averaged across observed listings

### Multi-Source Integration
When multiple sources track similar metrics:
- **Hackmageddon** provides incident counts by target country
- **Elsevier** provides academic research attention metrics
- **RSS Feeds** capture media and industry reporting
- **NVD** reflects formal vulnerability disclosures
- **Dark Web** indicates threat actor marketplace activity

These different perspectives should be analysed in combination for comprehensive threat intelligence.

### Geopolitical Context
- War/Conflict and Political Violence fields provide external context
- Holiday indicators may correlate with attack timing patterns
- Cross-referencing geopolitical events with cyber incidents enables attribution analysis

### Dark Web Data Interpretation
- Marketplace mentions indicate threat actor interest and capability availability
- Pricing reflects underground market supply/demand economics
- Sudden price changes may indicate law enforcement disruption or new exploit discovery
- Data represents observable marketplace activity, not comprehensive dark web coverage

### Google Trends Normalisation
- Values represent relative search interest (0-100 scale)
- 100 = peak search interest for that term across the entire dataset period
- 50 = half the search interest of the peak
- Trends indicate public awareness and concern, not necessarily attack volume

### Academic Literature Lag
- Elsevier mentions may lag real-world events due to publication cycles
- Increased academic attention often follows major incidents by 6-12 months
- Emerging threats may have low mention counts initially

---

## Recommended Potential Use Cases (For Future Work)

### 1. Threat Intelligence
- Identify emerging attack trends through multiple data sources
- Track threat evolution through dark web, academic, and incident data
- Correlate geopolitical events with cyber attack patterns

### 2. Predictive Modelling
- Forecast attack likelihood using temporal patterns
- Incorporate geopolitical indicators for conflict-related cyber activity
- Model seasonal effects using holiday indicators

### 3. Security Research
- Analyse gap between academic research focus and real-world threats
- Study lag between vulnerability disclosure and exploitation
- Investigate underground market economics and threat commoditisation

### 4. Policy Analysis
- Assess global cybersecurity posture by country
- Evaluate defence technology adoption through solution mentions
- Track international cyber conflict patterns

### 5. Risk Assessment
- Quantify country-specific threat exposure
- Benchmark attack volumes across industries and regions
- Identify high-risk periods through temporal pattern analysis

---

## Data Limitations

### Coverage Gaps
- Not all cyber incidents are publicly reported or captured
- Dark web data limited to monitored marketplaces (2020-2024)
- Academic literature has publication lag
- Media reporting may have geographic bias

### Attribution Challenges
- Attack attribution is inherently uncertain
- Country codes represent targets, not perpetrators
- "Unknown" and "?" categories capture attribution difficulties

### Temporal Considerations
- Monthly aggregation obscures intra-month patterns
- Some sources have limited historical data
- Real-time threat landscape extends beyond dataset end date

### Measurement Bias
- Google Trends reflects English-language search behaviour
- RSS feeds may over-represent Western media sources
- Dark web data represents observable marketplace activity only
- Academic literature may focus on novel rather than prevalent threats

---

## Version History

**Dataset Version**: 1.0  
**Dictionary Version**: 1.0  
**Last Updated**: December 2024  
**Compiled By**: Isobel Smith

---

## Contact & Citation

For questions regarding this dataset or data dictionary:
- Dataset documentation and methodology available on request


---

## Appendix A: Field Count Summary

| Category | Number of Fields |
|----------|------------------|
| Hackmageddon Attack Data | 1,036 fields (28 attack types × 37 countries) |
| Elsevier Attack Mentions | 52 fields |
| War/Conflict Data | 37 fields |
| Holiday Indicators | 37 fields |
| Elsevier PAT Mentions | 95 fields |
| ACLED Political Violence | 37 fields |
| Google Trends Attacks | 29 fields |
| Google Trends Solutions | 57 fields |
| University of Maryland | 444 fields (12 attack types × 37 countries) |
| Dark Web Monthly Counts | 67 fields |
| Dark Web Pricing | 68 fields |
| RSS Feed Attacks | 42 fields |
| RSS Feed Solutions | 119 fields |
| NVD Mentions | 40 fields |
| **Total Approximate Fields** | **~2,160 fields** |

---

## Appendix B: Acronyms & Abbreviations

| Acronym | Definition |
|---------|------------|
| ACLED | Armed Conflict Location & Event Data Project |
| APT | Advanced Persistent Threat |
| ATO | Account Takeover |
| BEC | Business Email Compromise |
| BIN | Bank Identification Number |
| C2 | Command and Control |
| CVE | Common Vulnerabilities and Exposures |
| CVV | Card Verification Value |
| DDoS | Distributed Denial of Service |
| DNS | Domain Name System |
| DoS | Denial of Service |
| EK | Exploit Kit |
| HTTPS | Hypertext Transfer Protocol Secure |
| IBE | Identity-Based Encryption |
| IDS | Intrusion Detection System |
| IoT | Internet of Things |
| IPS | Intrusion Prevention System |
| LFI | Local File Inclusion |
| LLM | Large Language Model |
| MaaS | Malware-as-a-Service |
| MITM | Man-in-the-Middle |
| ML | Machine Learning |
| DL | Deep Learning |
| NLP | Natural Language Processing |
| NVD | National Vulnerability Database |
| PAT | Pertinent Alleviation Technology |
| PKI | Public Key Infrastructure |
| RaaS | Ransomware-as-a-Service |
| RAT | Remote Access Trojan |
| RCE | Remote Code Execution |
| RFI | Remote File Inclusion |
| RRAM | Resistive Random-Access Memory |
| RSS | Really Simple Syndication |
| SIEM | Security Information and Event Management |
| SQL | Structured Query Language |
| SQLi | SQL Injection |
| SSL | Secure Sockets Layer |
| TLS | Transport Layer Security |
| USD | United States Dollar |
| VPN | Virtual Private Network |
| XSS | Cross-Site Scripting |

---

*End of Data Dictionary*
