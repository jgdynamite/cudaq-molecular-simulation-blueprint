# -----------------------------------------------------------------------------
# Provision a single Akamai Cloud Linode running on
# NVIDIA RTX PRO 6000 Blackwell Server Edition for cudaq-molecular-simulation-blueprint.
#
# This is intentionally minimal:
#
#   - One Linode (1-GPU plan) in a chosen region.
#   - Optional Linode firewall locking down the public surface to SSH + 8000.
#   - Optional rendering of an Ansible inventory for the follow-up
#     bootstrap_host.sh step that installs drivers, Docker, and the app.
#
# It does NOT bring up Kubernetes, multiple GPUs, multi-node setups, or any
# Akamai EdgeWorkers / Functions plumbing. Those are explicit non-goals for v1.
# -----------------------------------------------------------------------------

resource "random_password" "root" {
  length      = 32
  special     = true
  min_special = 4
  min_upper   = 4
  min_numeric = 4
}

resource "linode_instance" "blackwell" {
  label            = var.label
  region           = var.region
  type             = var.instance_type
  image            = var.image
  authorized_keys  = var.ssh_authorized_keys
  root_pass        = random_password.root.result
  private_ip       = var.private_ip
  disk_encryption  = var.disk_encryption
  tags             = var.tags
  watchdog_enabled = true

  metadata {
    user_data = base64encode(<<-EOT
      #cloud-config
      package_update: true
      package_upgrade: false
      packages:
        - curl
        - ca-certificates
        - gnupg
        - python3
        - python3-pip
      runcmd:
        - mkdir -p /var/log/cudaq-blueprint
        - echo "cudaq-blueprint host bootstrapped via cloud-init" > /var/log/cudaq-blueprint/bootstrap.log
    EOT
    )
  }
}

resource "linode_firewall" "blackwell" {
  count = var.create_firewall ? 1 : 0

  label = "${var.label}-fw"
  tags  = var.tags

  # The provider rejects an empty ipv4/ipv6 list, so a family with no CIDRs
  # must be omitted entirely (null) rather than passed as []. Restricting to
  # a single IPv4 /32 is the common case and used to fail the plan outright.
  inbound {
    label    = "allow-ssh"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "22"
    ipv4     = length(local.ssh_cidrs_v4) > 0 ? local.ssh_cidrs_v4 : null
    ipv6     = length(local.ssh_cidrs_v6) > 0 ? local.ssh_cidrs_v6 : null
  }

  inbound {
    label    = "allow-app"
    action   = "ACCEPT"
    protocol = "TCP"
    ports    = "8000"
    ipv4     = length(local.app_cidrs_v4) > 0 ? local.app_cidrs_v4 : null
    ipv6     = length(local.app_cidrs_v6) > 0 ? local.app_cidrs_v6 : null
  }

  inbound_policy  = "DROP"
  outbound_policy = "ACCEPT"

  linodes = [linode_instance.blackwell.id]
}

locals {
  blackwell_public_ip = try(tolist(linode_instance.blackwell.ipv4)[0], "")

  ssh_cidrs_v4 = [for c in var.firewall_allowed_ssh_cidrs : c if length(regexall(":", c)) == 0]
  ssh_cidrs_v6 = [for c in var.firewall_allowed_ssh_cidrs : c if length(regexall(":", c)) > 0]
  app_cidrs_v4 = [for c in var.firewall_allowed_app_cidrs : c if length(regexall(":", c)) == 0]
  app_cidrs_v6 = [for c in var.firewall_allowed_app_cidrs : c if length(regexall(":", c)) > 0]
}

resource "local_file" "ansible_inventory" {
  count    = var.render_ansible_inventory ? 1 : 0
  filename = "${path.module}/inventory.ini"
  content  = <<-EOT
    [blackwell]
    ${linode_instance.blackwell.label} ansible_host=${local.blackwell_public_ip} ansible_user=root

    [blackwell:vars]
    instance_type=${var.instance_type}
    region=${var.region}
  EOT
}
