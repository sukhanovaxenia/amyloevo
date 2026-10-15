#!/usr/bin/env Rscript
#
# Phylogenetic signal in APR block presence, and the regression on clade.
#
# Called by scripts/phylo_signal.py, which does the preparation it refuses to
# duplicate here: reconciling tips with traits, screening each trait for the
# prevalence D needs, and recording which identical tips carry different traits.
# This script receives only columns that already passed that screen.
#
# Why R. There is no Python implementation of Fritz & Purvis's D worth
# trusting, and writing one to avoid a dependency would put the least-tested
# code in the project at the point where the conclusion is drawn. caper,
# phytools and phylolm are the reference implementations these statistics are
# defined by.
#
# Usage:
#   Rscript phylo_signal.R <tree.nwk> <traits.tsv> <covariates.tsv>
#                          <outgroup.txt> <out.tsv> <signal> <regression>
#                          <permutations> <seed>
#
# traits.tsv:     id + one column per block, 0/1 (for D) -- the screened set
# covariates.tsv: id, group, length, root_distance, occupancy_<block> (continuous)
# outgroup.txt:   one tip label per line; empty file means do not reroot

suppressPackageStartupMessages({
  missing <- character(0)
  for (pkg in c("ape", "caper", "phytools", "phylolm")) {
    if (!requireNamespace(pkg, quietly = TRUE)) missing <- c(missing, pkg)
  }
  if (length(missing)) {
    stop(sprintf(
      paste0("missing R package(s): %s\n",
             "  install.packages(c(%s))\n",
             "caper supplies Fritz & Purvis D, phytools Pagel's lambda, ",
             "phylolm the phylogenetic logistic regression. None of the three ",
             "is substitutable by a correlation on the tips."),
      paste(missing, collapse = ", "),
      paste(sprintf('"%s"', missing), collapse = ", ")
    ), call. = FALSE)
  }
  library(ape); library(caper); library(phytools); library(phylolm)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 9) stop("expected 9 arguments; see the header", call. = FALSE)
tree_file <- args[1]; traits_file <- args[2]; covar_file <- args[3]
outgroup_file <- args[4]; out_file <- args[5]
signal <- args[6]; regression <- args[7]
permutations <- as.integer(args[8]); seed <- as.integer(args[9])
set.seed(seed)

phy <- read.tree(tree_file)
traits <- read.delim(traits_file, stringsAsFactors = FALSE, check.names = FALSE)
covar <- read.delim(covar_file, stringsAsFactors = FALSE, check.names = FALSE)

# --- rooting ---------------------------------------------------------------
# ape::root with resolve.root = TRUE puts the root ON the stem of the outgroup
# and leaves a bifurcating root. Both matter: is.rooted() is FALSE for a root
# of degree three, and phyloglm builds its covariance from root-to-tip paths,
# so it refuses an unrooted tree. Python has already checked that the outgroup
# is monophyletic here; this is the placement, not the decision.
outgroup <- character(0)
if (file.exists(outgroup_file)) {
  outgroup <- readLines(outgroup_file, warn = FALSE)
  outgroup <- outgroup[nzchar(outgroup)]
}
if (length(outgroup)) {
  outgroup <- intersect(outgroup, phy$tip.label)
  phy <- root(phy, outgroup = outgroup, resolve.root = TRUE, edgelabel = TRUE)
  message(sprintf("rooted on %d outgroup tip(s); is.rooted = %s",
                  length(outgroup), is.rooted(phy)))
}
if (!is.rooted(phy)) {
  stop(paste0("the tree is not rooted and no usable outgroup was given. D and ",
              "lambda both read ancestry off the branching order, so an ",
              "unrooted tree gives them nothing to read."), call. = FALSE)
}
# Zero-length branches are not an error -- identical sequences produce them --
# but they are where a trait change has no time under it, so count them.
n_zero <- sum(phy$edge.length <= 0, na.rm = TRUE)
if (n_zero) {
  message(sprintf("%d of %d branch(es) have length <= 0", n_zero,
                  length(phy$edge.length)))
  # phyloglm's transformations divide by branch length; a hard zero makes the
  # covariance singular. The usual remedy, and the one made explicit here.
  phy$edge.length[phy$edge.length <= 0] <- 1e-8
}

blocks <- setdiff(names(traits), "id")
rows <- list()

for (blk in blocks) {
  frame <- data.frame(
    id = traits$id,
    presence = as.integer(traits[[blk]]),
    stringsAsFactors = FALSE
  )
  occ_col <- paste0("occupancy_", blk)
  frame$occupancy <- if (occ_col %in% names(covar)) {
    covar[[occ_col]][match(frame$id, covar$id)]
  } else NA_real_
  frame$group <- covar$group[match(frame$id, covar$id)]
  frame$length <- covar$length[match(frame$id, covar$id)]
  frame$root_distance <- covar$root_distance[match(frame$id, covar$id)]
  frame <- frame[frame$id %in% phy$tip.label, , drop = FALSE]

  row <- list(
    block = blk, n = nrow(frame), n_present = sum(frame$presence),
    d_estimate = NA_real_, d_p_random = NA_real_, d_p_brownian = NA_real_,
    lambda = NA_real_, lambda_p = NA_real_,
    glm_term = NA_character_, glm_estimate = NA_real_, glm_p = NA_real_,
    glm_alpha = NA_real_, note = ""
  )
  notes <- character(0)

  # --- Fritz & Purvis D --------------------------------------------------
  # D = 0 is the Brownian expectation (the trait follows the tree as a
  # continuous character thresholded once), D = 1 is phylogenetically random.
  # Both p-values are needed: a D near 0.5 can be significantly different from
  # random AND from Brownian, which is the informative case and the one a
  # single p-value hides.
  if (signal %in% c("both", "fritz_purvis_d")) {
    d <- try({
      cdata <- comparative.data(phy, frame, names.col = "id", vcv = FALSE,
                                warn.dropped = FALSE)
      phylo.d(cdata, binvar = presence, permut = permutations)
    }, silent = TRUE)
    if (inherits(d, "try-error")) {
      notes <- c(notes, paste0("phylo.d failed: ",
                               trimws(as.character(attr(d, "condition")$message))))
    } else {
      row$d_estimate <- as.numeric(d$DEstimate)
      row$d_p_random <- as.numeric(d$Pval1)     # H0: random association
      row$d_p_brownian <- as.numeric(d$Pval0)   # H0: Brownian
    }
  }

  # --- Pagel's lambda on the continuous occupancy -------------------------
  # On the occupancy fraction, not on the binarised column: lambda is defined
  # for a continuous character, and fitting it to 0/1 asks a Brownian model to
  # describe a two-point distribution. Branch lengths here are substitutions
  # per site, not time, so lambda is scaled by rate as much as by history --
  # read it as "does the trait follow this tree", not as a clock.
  if (signal %in% c("both", "pagel_lambda") && any(is.finite(frame$occupancy))) {
    x <- frame$occupancy
    names(x) <- frame$id
    x <- x[is.finite(x)]
    lam <- try(phylosig(keep.tip(phy, names(x)), x, method = "lambda",
                        test = TRUE), silent = TRUE)
    if (inherits(lam, "try-error")) {
      notes <- c(notes, paste0("phylosig failed: ",
                               trimws(as.character(attr(lam, "condition")$message))))
    } else {
      row$lambda <- as.numeric(lam$lambda)
      row$lambda_p <- as.numeric(lam$P)
    }
  }

  # --- phylogenetic logistic regression ----------------------------------
  # Fitted on `length` because that one has replication across the tree: a
  # longer protein has more room for an APR and the effect must be controlled
  # before any clade effect is believed.
  #
  # The clade term is NOT fitted here, and the omission is the point. Each
  # paralogue group is one monophyletic clade descending from one duplication,
  # so a group contrast has two independent events behind it however many tips
  # it spans. phyloglm would return a coefficient and a p-value computed as if
  # there were 372. The controlled instrument for the paralogue difference is
  # the paired within-taxon contrast over taxa that carry all three copies --
  # 82 of them here -- and that is a paired test, not a regression.
  if (regression == "phyloglm" && any(is.finite(frame$length))) {
    sub <- frame[is.finite(frame$length) & frame$id %in% phy$tip.label, ]
    if (length(unique(sub$presence)) < 2) {
      notes <- c(notes, "phyloglm skipped: presence is constant on the fitted subset")
    } else {
      rownames(sub) <- sub$id
      fit <- try(phyloglm(presence ~ length, data = sub,
                          phy = keep.tip(phy, sub$id),
                          method = "logistic_MPLE", btol = 50), silent = TRUE)
      if (inherits(fit, "try-error")) {
        notes <- c(notes, paste0("phyloglm failed: ",
                                 trimws(as.character(attr(fit, "condition")$message))))
      } else {
        s <- summary(fit)$coefficients
        if ("length" %in% rownames(s)) {
          row$glm_term <- "length"
          row$glm_estimate <- s["length", "Estimate"]
          row$glm_p <- s["length", "p.value"]
        }
        row$glm_alpha <- as.numeric(fit$alpha)
      }
    }
  }

  row$note <- paste(notes, collapse = "; ")
  rows[[blk]] <- as.data.frame(row, stringsAsFactors = FALSE)
}

out <- if (length(rows)) do.call(rbind, rows) else
  data.frame(block = character(0))
write.table(out, out_file, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "")
message(sprintf("wrote %d row(s) to %s", nrow(out), out_file))
