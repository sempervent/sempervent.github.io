# Building a ZFS Tank with OS on NVMe

**Objective**: Build a production-ready ZFS storage system with OS on fast NVMe and data on a resilient ZFS pool. Transform your storage from fragile to bulletproof.

Why NVMe for OS? Speed + separation of concerns. Why ZFS tank? Snapshots, checksums, compression, resilience. This is how you build storage that doesn't die when drives fail.

## 1) Hardware Prep: The Foundation

### Minimum Requirements

```bash
# OS Drive (NVMe)
- 500GB+ NVMe SSD (e.g., Samsung 980 Pro, WD Black SN850)
- PCIe 3.0 x4 or PCIe 4.0 x4 slot
- M.2 2280 form factor

# ZFS Pool Drives (2-6 drives)
- 2TB+ HDDs or SSDs (same size for optimal performance)
- SATA 6Gb/s or SAS 12Gb/s
- Consider enterprise drives for 24/7 operation

# System Requirements
- 8GB RAM minimum, 16GB+ recommended for ARC cache
- Dedicated SATA/SAS controller (optional but recommended)
- UPS (ZFS doesn't like sudden power loss)
```

**Why This Hardware**: NVMe gives you fast boot and system operations. ZFS pool gives you data integrity and performance. Separation prevents OS issues from affecting data.

### Optional Performance Add-Ons

```bash
# L2ARC Cache (optional)
- 500GB+ NVMe SSD for read cache
- Separate from OS drive

# SLOG (optional)
- High-endurance NVMe for sync write log
- Optane or enterprise NVMe preferred

# Backup Drives
- External USB 3.0+ drives for backups
- Network storage for remote replication
```

**Why These Add-Ons**: L2ARC speeds up reads, SLOG speeds up sync writes. Backups are non-negotiable.

## 2) Install OS on NVMe: The Fast Foundation

### Boot Installer and Target NVMe

```bash
# Boot from Ubuntu Server 24.04 LTS or Debian 12
# Select "Manual partitioning" during installation
# Target the NVMe device (e.g., /dev/nvme0n1)
```

**Why Manual Partitioning**: We need precise control over the layout. Automatic partitioning doesn't understand our ZFS plans.

### Partition Layout (GPT)

```bash
# Example partitioning for 1TB NVMe:
# Partition 1: EFI System Partition (512MB)
# Partition 2: Boot partition (2GB, ext4)
# Partition 3: Root partition (rest, ext4 or ZFS-root)

# Using gdisk or parted:
sudo gdisk /dev/nvme0n1

# Create GPT table
o
y

# Create EFI partition
n
1
+512M
ef00

# Create boot partition
n
2
+2G
8300

# Create root partition
n
3
+100G
8300

# Write changes
w
y
```

**Why This Layout**: EFI for UEFI boot, boot partition for kernel/initrd, root for everything else. Separation prevents boot issues from affecting data.

### Format and Mount

```bash
# Format partitions
sudo mkfs.fat -F32 /dev/nvme0n1p1  # EFI
sudo mkfs.ext4 /dev/nvme0n1p2      # Boot
sudo mkfs.ext4 /dev/nvme0n1p3      # Root

# Mount for installation
sudo mount /dev/nvme0n1p3 /mnt
sudo mkdir /mnt/boot
sudo mount /dev/nvme0n1p2 /mnt/boot
sudo mkdir /mnt/boot/efi
sudo mount /dev/nvme0n1p1 /mnt/boot/efi
```

**Why These Formats**: FAT32 for EFI compatibility, ext4 for reliability and performance. ZFS-root is advanced and not recommended for beginners.

### Complete OS Installation

```bash
# Install base system
sudo debootstrap jammy /mnt
sudo chroot /mnt /bin/bash

# Install kernel and essential packages
apt update
apt install -y linux-image-generic linux-headers-generic
apt install -y grub-efi-amd64 efibootmgr
apt install -y zfsutils-linux

# Configure bootloader
grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=ubuntu
update-grub

# Exit chroot and reboot
exit
sudo umount -R /mnt
sudo reboot
```

**Why This Process**: Standard Ubuntu installation with ZFS tools pre-installed. We're building the foundation for our storage system.

## 3) Install ZFS: The Storage Engine

### Install ZFS Tools

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install ZFS
sudo apt install -y zfsutils-linux zfs-initramfs

# Add ZFS to initramfs
sudo update-initramfs -u

# Reboot to ensure ZFS modules load
sudo reboot
```

**Why ZFS Tools**: We need the full ZFS stack. The initramfs update ensures ZFS pools can be imported during boot.

### Verify ZFS Installation

```bash
# Check ZFS version
zfs version

# Check available modules
lsmod | grep zfs

# Check if ZFS is ready
sudo zpool list
```

**Why This Verification**: ZFS must be working before we create pools. Broken ZFS means broken storage.

## 4) Build the Tank: The Storage Pool

### Identify Available Drives

```bash
# List all block devices
lsblk

# Identify drives for ZFS pool (avoid OS drive!)
# Example: /dev/sdb, /dev/sdc, /dev/sdd, /dev/sde

# Check drive health
sudo smartctl -a /dev/sdb
sudo smartctl -a /dev/sdc
sudo smartctl -a /dev/sdd
sudo smartctl -a /dev/sde
```

**Why Drive Health Check**: Dead drives kill pools. Check before you commit.

### Create ZFS Pool

#### RAIDZ1 (4 drives, 1 parity)

```bash
# Create RAIDZ1 pool (3 data + 1 parity)
sudo zpool create -f tank raidz1 /dev/sdb /dev/sdc /dev/sdd /dev/sde

# Verify pool creation
sudo zpool status
sudo zpool list
```

#### Mirror (2 drives, 1:1 redundancy)

```bash
# Create mirror pool (1:1 redundancy)
sudo zpool create -f tank mirror /dev/sdb /dev/sdc

# Verify pool creation
sudo zpool status
sudo zpool list
```

**Why These Configurations**: RAIDZ1 gives you 3+1 redundancy with good performance. Mirror gives you 1:1 redundancy with maximum performance.

### Pool Layout Diagram

```mermaid
graph TB
    subgraph "System Architecture"
        subgraph "NVMe OS Drive"
            EFI[EFI Partition<br/>512MB]
            BOOT[Boot Partition<br/>2GB ext4]
            ROOT[Root Partition<br/>100GB ext4]
        end
        
        subgraph "ZFS Tank Pool"
            POOL[ZFS Pool 'tank']
            subgraph "RAIDZ1 Configuration"
                SDB[/dev/sdb<br/>Data]
                SDC[/dev/sdc<br/>Data]
                SDD[/dev/sdd<br/>Data]
                SDE[/dev/sde<br/>Parity]
            end
        end
        
        subgraph "Datasets"
            DATA[tank/data<br/>Compression: zstd]
            MEDIA[tank/media<br/>Recordsize: 1M]
            VM[tank/vm<br/>Recordsize: 128K]
        end
    end
    
    %% Connections
    EFI --> BOOT
    BOOT --> ROOT
    ROOT --> POOL
    POOL --> SDB
    POOL --> SDC
    POOL --> SDD
    POOL --> SDE
    POOL --> DATA
    POOL --> MEDIA
    POOL --> VM
    
    %% Styling
    classDef nvme fill:#ff6b6b,stroke:#d63031,stroke-width:3px
    classDef zfs fill:#74b9ff,stroke:#0984e3,stroke-width:2px
    classDef dataset fill:#00b894,stroke:#00a085,stroke-width:2px
    classDef drive fill:#fdcb6e,stroke:#e17055,stroke-width:2px
    
    class EFI,BOOT,ROOT nvme
    class POOL zfs
    class DATA,MEDIA,VM dataset
    class SDB,SDC,SDD,SDE drive
```

**Why This Layout**: Clear separation between OS and data. ZFS pool provides redundancy and performance. Datasets organize data by use case.

## 5) Verify Pool: The Health Check

### Check Pool Status

```bash
# Detailed pool status
sudo zpool status -v

# Pool information
sudo zpool list -v

# Pool health
sudo zpool health tank
```

**Why These Checks**: Pool status shows redundancy, health shows problems. Monitor regularly.

### Check Pool Properties

```bash
# Show all pool properties
sudo zpool get all tank

# Show specific properties
sudo zpool get compression tank
sudo zpool get checksum tank
sudo zpool get redundancy tank
```

**Why Property Monitoring**: ZFS properties control behavior. Understanding them prevents surprises.

## 6) Create Datasets: The Organization

### Create Base Datasets

```bash
# Create data dataset with compression
sudo zfs create -o compression=zstd -o atime=off tank/data

# Create media dataset with large recordsize
sudo zfs create -o compression=zstd -o recordsize=1M tank/media

# Create VM dataset with small recordsize
sudo zfs create -o compression=zstd -o recordsize=128K tank/vm

# Create backup dataset
sudo zfs create -o compression=zstd tank/backup
```

**Why These Settings**: Compression saves space, recordsize matches workload, atime=off improves performance.

### Set Dataset Properties

```bash
# Set quota on data dataset
sudo zfs set quota=100G tank/data

# Set reservation on media dataset
sudo zfs set reservation=500G tank/media

# Set compression on VM dataset
sudo zfs set compression=zstd tank/vm

# Set snapdir on backup dataset
sudo zfs set snapdir=visible tank/backup
```

**Why These Properties**: Quotas prevent runaway growth, reservations guarantee space, snapdir makes snapshots visible.

### Verify Dataset Creation

```bash
# List all datasets
sudo zfs list -o name,mountpoint,used,available,compression

# Show dataset properties
sudo zfs get all tank/data

# Check mount points
sudo zfs list -o name,mountpoint
```

**Why This Verification**: Datasets must be mounted and accessible. Properties must be set correctly.

## 7) Mounting & Persistence: The Boot Process

### Automatic Mounting

```bash
# ZFS mounts datasets automatically via /etc/zfs/zfs-list.cache
# Check if datasets are mounted
sudo zfs list -o name,mountpoint

# Manually mount if needed
sudo zfs mount tank/data
sudo zfs mount tank/media
sudo zfs mount tank/vm
```

**Why Automatic Mounting**: ZFS handles mounting automatically. Manual mounting is only needed for troubleshooting.

### Verify Mount Points

```bash
# Check mount points
df -h | grep tank

# Check dataset mount status
sudo zfs list -o name,mountpoint,mounted

# Test write access
sudo touch /tank/data/test.txt
sudo rm /tank/data/test.txt
```

**Why Mount Verification**: Datasets must be accessible for use. Write tests confirm permissions.

## 8) Tuning & Best Practices: The Performance

### Compression Settings

```bash
# Set compression on all datasets
sudo zfs set compression=zstd tank/data
sudo zfs set compression=zstd tank/media
sudo zfs set compression=zstd tank/vm

# Check compression ratio
sudo zfs list -o name,used,compressratio
```

**Why Compression**: zstd provides excellent compression with minimal CPU overhead. Saves space and improves I/O.

### Recordsize Optimization

```bash
# Set recordsize for different workloads
sudo zfs set recordsize=1M tank/media    # Large files
sudo zfs set recordsize=128K tank/vm     # VM images
sudo zfs set recordsize=16K tank/data    # Database files
```

**Why Recordsize**: Matches workload to storage block size. Improves performance and reduces fragmentation.

### ARC Cache Tuning

```bash
# Check current ARC size
cat /proc/spl/kstat/zfs/arcstats | grep size

# Set ARC size limit (50% of RAM)
echo 8589934592 | sudo tee /sys/module/zfs/parameters/zfs_arc_max

# Make ARC limit persistent
echo 'options zfs zfs_arc_max=8589934592' | sudo tee -a /etc/modprobe.d/zfs.conf
```

**Why ARC Tuning**: ARC cache improves read performance. Limit prevents memory starvation.

### Snapshot Management

```bash
# Create snapshots
sudo zfs snapshot tank/data@baseline
sudo zfs snapshot tank/media@baseline
sudo zfs snapshot tank/vm@baseline

# List snapshots
sudo zfs list -t snapshot

# Create recursive snapshots
sudo zfs snapshot -r tank@daily-$(date +%Y%m%d)
```

**Why Snapshots**: Point-in-time recovery, backup source, rollback capability. Essential for data protection.

## 9) Optional Add-Ons: The Performance Boosters

### L2ARC Cache (Read Cache)

```bash
# Add L2ARC cache (requires spare SSD)
sudo zpool add tank cache /dev/nvme1n1

# Check L2ARC status
sudo zpool status tank
sudo zfs list -o name,l2arc_hits,l2arc_misses
```

**Why L2ARC**: Extends ARC cache to disk. Improves read performance for large datasets.

### SLOG (Sync Write Log)

```bash
# Add SLOG (requires high-endurance SSD)
sudo zpool add tank log /dev/nvme2n1

# Check SLOG status
sudo zpool status tank
sudo zfs list -o name,sync
```

**Why SLOG**: Accelerates sync writes. Essential for databases and VMs.

### Automated Scrubbing

```bash
# Create scrub script
sudo tee /usr/local/bin/zfs-scrub.sh << 'EOF'
#!/bin/bash
zpool scrub tank
EOF

sudo chmod +x /usr/local/bin/zfs-scrub.sh

# Add to crontab (weekly scrub)
echo "0 2 * * 0 /usr/local/bin/zfs-scrub.sh" | sudo crontab -
```

**Why Scrubbing**: Detects and repairs data corruption. Weekly scrubs maintain data integrity.

## 10) Backup & Safety: The Disaster Recovery

### Pool Export/Import

```bash
# Export pool (for maintenance)
sudo zpool export tank

# Import pool
sudo zpool import tank

# Import with specific name
sudo zpool import -f tank backup-tank
```

**Why Export/Import**: Safe pool maintenance, pool migration, disaster recovery. Practice these procedures.

### Send/Receive Backup

```bash
# Create backup pool
sudo zpool create backup-pool mirror /dev/sdf /dev/sdg

# Send dataset to backup
sudo zfs send tank/data@baseline | sudo zfs recv backup-pool/data

# Incremental backup
sudo zfs send -i tank/data@baseline tank/data@daily | sudo zfs recv backup-pool/data
```

**Why Send/Receive**: Efficient incremental backups. Only changed data is transferred.

### Configuration Backup

```bash
# Backup pool configuration
sudo zpool status > /home/user/zfs-backup/pool-status.txt
sudo zfs list -o name,mountpoint,used,available > /home/user/zfs-backup/dataset-list.txt
sudo zfs get all tank > /home/user/zfs-backup/pool-properties.txt

# Store in Git
cd /home/user/zfs-backup
git init
git add .
git commit -m "ZFS configuration backup"
```

**Why Configuration Backup**: Pool recreation requires exact configuration. Git provides version control.

## 11) Monitoring & Maintenance: The Operations

### Health Monitoring

```bash
# Check pool health
sudo zpool status -v

# Check drive health
sudo smartctl -a /dev/sdb | grep -E "(Reallocated|Pending|Uncorrectable)"

# Check pool errors
sudo zpool status tank | grep -E "(errors|corruption)"
```

**Why Health Monitoring**: Early problem detection prevents data loss. Regular monitoring is essential.

### Performance Monitoring

```bash
# Check ARC hit ratio
cat /proc/spl/kstat/zfs/arcstats | grep -E "(hits|misses)"

# Check compression ratio
sudo zfs list -o name,used,compressratio

# Check I/O statistics
sudo zfs list -o name,read,write
```

**Why Performance Monitoring**: Identifies bottlenecks and optimization opportunities. Essential for tuning.

### Automated Maintenance

```bash
# Create maintenance script
sudo tee /usr/local/bin/zfs-maintenance.sh << 'EOF'
#!/bin/bash
# Weekly scrub
zpool scrub tank

# Check pool health
zpool status tank

# Update pool properties
zfs set compression=zstd tank/data
zfs set compression=zstd tank/media
zfs set compression=zstd tank/vm
EOF

sudo chmod +x /usr/local/bin/zfs-maintenance.sh

# Add to crontab
echo "0 2 * * 0 /usr/local/bin/zfs-maintenance.sh" | sudo crontab -
```

**Why Automated Maintenance**: Consistent maintenance prevents problems. Automation reduces human error.

## 12) Troubleshooting: When Things Go Wrong

### Common Issues

```bash
# Pool won't import
sudo zpool import -f tank

# Dataset won't mount
sudo zfs mount tank/data

# Drive failure
sudo zpool offline tank /dev/sdb
sudo zpool replace tank /dev/sdb /dev/sdf

# Pool corruption
sudo zpool scrub tank
sudo zpool clear tank
```

**Why These Solutions**: Common problems have common solutions. Practice these procedures.

### Recovery Procedures

```bash# Emergency pool import
sudo zpool import -f -d /dev/disk/by-id tank

# Recover from backup
sudo zpool import backup-pool
sudo zfs send backup-pool/data@latest | sudo zfs recv tank/data

# Rebuild from scratch
sudo zpool create -f tank raidz1 /dev/sdb /dev/sdc /dev/sdd /dev/sde
sudo zfs create tank/data
sudo zfs create tank/media
sudo zfs create tank/vm
```

**Why Recovery Procedures**: Disasters happen. Recovery procedures restore service quickly.

## 13) TL;DR Quickstart

```bash
# 1. Install OS on NVMe
# - Boot installer, target NVMe device
# - Create partitions: EFI (512MB), Boot (2GB), Root (rest)
# - Install Ubuntu/Debian with ZFS tools

# 2. Install ZFS
sudo apt install -y zfsutils-linux zfs-initramfs
sudo update-initramfs -u
sudo reboot

# 3. Create ZFS pool
sudo zpool create -f tank raidz1 /dev/sdb /dev/sdc /dev/sdd /dev/sde

# 4. Create datasets
sudo zfs create -o compression=zstd tank/data
sudo zfs create -o compression=zstd tank/media
sudo zfs create -o compression=zstd tank/vm

# 5. Verify setup
sudo zpool status
sudo zfs list
df -h | grep tank

# 6. Create snapshots
sudo zfs snapshot tank/data@baseline
sudo zfs snapshot tank/media@baseline
sudo zfs snapshot tank/vm@baseline

# 7. Set up monitoring
sudo zpool scrub tank
sudo zfs list -o name,used,compressratio
```

## 14) The Machine's Summary

ZFS transforms your storage from fragile to bulletproof. With OS on NVMe and data on ZFS, you get speed, reliability, and advanced features that traditional filesystems can't match.

**The Dark Truth**: ZFS is complex. Misconfiguration kills data. But with proper planning and practice, it becomes the most reliable storage system you'll ever use.

**The Machine's Mantra**: "In redundancy we trust, in snapshots we recover, and in the ZFS tank we find the path to storage immortality."

**Why This Matters**: Data is valuable. ZFS protects it with checksums, snapshots, and redundancy. When drives fail, ZFS keeps your data alive.

---

*This tutorial provides the complete machinery for building a production-ready ZFS storage system. The tank scales from terabytes to petabytes, from home labs to data centers.*
