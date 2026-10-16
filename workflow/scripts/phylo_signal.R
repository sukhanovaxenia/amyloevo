#!/usr/bin/env Rscript
#
# Phylogenetic signal in APR block presence, and the regression on length.
#
# Called by scripts/phylo_signal.py, which does the preparation it refuses to
# duplicate here: reconciling tips with traits, screening every trait for the
# prevalence D needs -- over the whole tree and inside each clade separately --
# and recording which identical tips carry different traits. This script
# receives a work list of (scope, block) pairs that already passed that screen
# and does not re-decide any of it.
#
# Why R. There is no Python implementation of Fritz & Purvis's D worth
# trusting, and writing one to avoid a dependency would put the least-tested
# code in the project at the point where the conclusion is drawn. caper,
# phytools and phylolm are the reference implementations these statistics are
# defined by.
#
# Usage:
#   Rscript phylo_signal.R <tree.nwk> <traits.tsv> <covariates.tsv> <tasks.tsv>
#                          <outgroup.txt> <out.tsv> <signal> <regression>
#                          <permutations> <seed>
#
# traits.tsv:     id + one column per block, 0/1
# covariates.tsv: id, group, length, root_distance, occupancy_<block>
# tasks.tsv:      scope, block -- scope is "all" or a clade name from `group`
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
if (length(args) < 10) stop("expected 10 arguments; see the header", call. = FALSE)
tree_file <- args[1]; traits_file <- args[2]; covar_file <- args[3]
tasks_file <- args[4]; outgroup_file <- args[5]; out_file <- args[6]
signal <- args[7]; regression <- args[8]
permutations <- as.integer(args[9]); seed <- as.integer(args[10])
set.seed(seed)

phy <- read.tree(tree_file)
traits <- read.delim(traits_file, stringsAsFactors = FALSE, check.names = FALSE)
covar <- read.delim(covar_file, stringsAsFactors = FALSE, check.names = FALSE)
tasks <- read.delim(tasks_file, stringsAsFactors = FALSE, check.names = FALSE)

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

# --- node labels have to go ------------------------------------------------
# caper's comparative.data refuses a tree whose labels are "duplicated between
# tips and nodes", and it checks `duplicated(c(tip.label, node.label))` -- one
# combined vector. IQ-TREE writes the branch support into the internal node
# labels, so any support value occurring twice is a duplicate by that test.
# On the FXR1 tree 126 of 330 node labels are repeated ("100/100" eleven
# times), and every single phylo.d call failed with that message while lambda
# and phyloglm, which never look at node labels, ran fine. The result was a
# table with a complete D column of nothing and no indication that the cause
# was a label check rather than the data.
#
# The support values are not needed here: they are read off the treefile on the
# Python side and reported there. So they are dropped, loudly.
if (!is.null(phy$node.label)) {
  n_dup <- sum(duplicated(c(phy$tip.label, phy$node.label)))
  message(sprintf(
    "dropping %d internal node label(s) before comparative.data (%d of them %s)",
    length(phy$node.label), n_dup,
    "repeated, which caper rejects as duplicated labels)"
  ))
  phy$node.label <- NULL
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

rows <- list()

for (k in seq_len(nrow(tasks))) {
  scope <- tasks$scope[k]
  blk <- tasks$block[k]

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

  # A clade scope is that clade's tips and nothing else. The subtree is taken
  # by keep.tip rather than by the clade's MRCA: Python has already checked
  # that the group is monophyletic, and if it were not, the MRCA would quietly
  # include the other groups' sequences.
  if (scope != "all") {
    frame <- frame[!is.na(frame$group) & frame$group == scope, , drop = FALSE]
  }
  if (nrow(frame) < 4) {
    rows[[length(rows) + 1]] <- data.frame(
      scope = scope, block = blk, n = nrow(frame), n_present = sum(frame$presence),
      d_estimate = NA_real_, d_p_random = NA_real_, d_p_brownian = NA_real_,
      pagel_lambda = NA_real_, lambda_p = NA_real_, glm_term = NA_character_,
      glm_estimate = NA_real_, glm_p = NA_real_, glm_alpha = NA_real_,
      glm_alpha_bound = NA_integer_, glm_converged = NA_integer_,
      note = "fewer than 4 tips in this scope", stringsAsFactors = FALSE
    )
    next
  }
  sub_phy <- if (scope == "all") phy else keep.tip(phy, frame$id)

  row <- list(
    scope = scope, block = blk, n = nrow(frame),
    n_present = sum(frame$presence),
    d_estimate = NA_real_, d_p_random = NA_real_, d_p_brownian = NA_real_,
    pagel_lambda = NA_real_, lambda_p = NA_real_,
    glm_term = NA_character_, glm_estimate = NA_real_, glm_p = NA_real_,
    glm_alpha = NA_real_, glm_alpha_bound = NA_integer_,
    glm_converged = NA_integer_, note = ""
  )
  notes <- character(0)

  # --- Fritz & Purvis D --------------------------------------------------
  # D rescales the observed sum of sister-clade differences against two
  # simulated nulls: 0 is the mean under a Brownian threshold model (the trait
  # behaves like a continuous character evolving on this tree, cut once at the
  # observed prevalence), 1 is the mean under random shuffling of the tip
  # values. Both p-values are needed: a D near the middle can be significantly
  # different from random AND from Brownian, which is the informative case and
  # the one a single p hides.
  if (signal %in% c("both", "fritz_purvis_d")) {
    d <- try({
      cdata <- comparative.data(sub_phy, frame, names.col = "id", vcv = FALSE,
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
  # Lambda multiplies the off-diagonal (shared-ancestry) entries of the
  # phylogenetic covariance matrix: 1 leaves the tree as given, so the trait
  # is Brownian on it, and 0 flattens it to a star, so the tips are
  # independent. It is fitted by ML and tested by a likelihood ratio against
  # lambda = 0.
  #
  # On the continuous occupancy fraction, not on the binarised column: lambda
  # is defined for a continuous character, and fitting it to 0/1 asks a
  # Brownian model to describe a two-point distribution. Branch lengths here
  # are substitutions per site, not time, so lambda is scaled by rate as much
  # as by history -- read it as "does the trait follow this tree", not as a
  # clock.
  if (signal %in% c("both", "pagel_lambda") && any(is.finite(frame$occupancy))) {
    x <- frame$occupancy
    names(x) <- frame$id
    x <- x[is.finite(x)]
    lam <- try(phylosig(keep.tip(sub_phy, names(x)), x, method = "lambda",
                        test = TRUE), silent = TRUE)
    if (inherits(lam, "try-error")) {
      notes <- c(notes, paste0("phylosig failed: ",
                               trimws(as.character(attr(lam, "condition")$message))))
    } else {
      row$pagel_lambda <- as.numeric(lam$lambda)
      row$lambda_p <- as.numeric(lam$P)
    }
  }

  # --- phylogenetic logistic regression ----------------------------------
  # Fitted on `length` because that one has replication across the tree: a
  # longer protein has more room for an APR and the effect must be controlled
  # before any clade effect is believed.
  #
  # The clade term is NOT fitted, and the omission is the point. Each paralogue
  # group is one monophyletic clade descending from one duplication, so a group
  # contrast has two independent events behind it however many tips it spans.
  # phyloglm would return a coefficient and a p-value computed as if there were
  # 282. The controlled instrument for the paralogue difference is the paired
  # within-taxon contrast over taxa carrying all three copies -- 55 of them
  # here, once the completeness filter is applied; 87 of 104 taxa carry all
  # three before it -- and that is a paired test, not a regression.
  #
  # alphaWarn and convergence are recorded, not just alpha. alpha is phyloglm's
  # phylogenetic-correlation parameter and it is searched over a bounded
  # interval; an estimate sitting on a bound is not an estimate, and on the
  # first FXR1 run all 15 blocks returned alpha = 1.51447 to six figures, which
  # is one value found fifteen times rather than fifteen measurements.
  if (regression == "phyloglm" && any(is.finite(frame$length))) {
    sub <- frame[is.finite(frame$length), , drop = FALSE]
    if (length(unique(sub$presence)) < 2) {
      notes <- c(notes, "phyloglm skipped: presence is constant on the fitted subset")
    } else {
      rownames(sub) <- sub$id
      # btol bounds how far the fitted probabilities may run from the data.
      # 50 is generous and on small, deep subtrees it lets the branch-length
      # transformation produce NaN ("edge.length[i] is NaN. Please reduce
      # btol"). phylolm says to reduce it, so that is what happens here rather
      # than the fit being lost: two of the FXR1 per-clade fits died this way
      # on the first run. A retry at a smaller bound is recorded in the note,
      # because the two fits are not interchangeable.
      sub_tree <- keep.tip(sub_phy, sub$id)
      fit <- try(phyloglm(presence ~ length, data = sub, phy = sub_tree,
                          method = "logistic_MPLE", btol = 50), silent = TRUE)
      if (inherits(fit, "try-error")) {
        first <- trimws(as.character(attr(fit, "condition")$message))
        fit <- try(phyloglm(presence ~ length, data = sub, phy = sub_tree,
                            method = "logistic_MPLE", btol = 10), silent = TRUE)
        if (!inherits(fit, "try-error")) {
          notes <- c(notes, "phyloglm refitted at btol = 10 after btol = 50 failed")
        }
      } else {
        first <- ""
      }
      if (inherits(fit, "try-error")) {
        notes <- c(notes, paste0(
          "phyloglm failed at btol = 50 and 10: ", first))
      } else {
        s <- summary(fit)$coefficients
        if ("length" %in% rownames(s)) {
          row$glm_term <- "length"
          row$glm_estimate <- s["length", "Estimate"]
          row$glm_p <- s["length", "p.value"]
        }
        row$glm_alpha <- as.numeric(fit$alpha)
        if (!is.null(fit$alphaWarn)) {
          row$glm_alpha_bound <- as.integer(fit$alphaWarn)
          if (isTRUE(fit$alphaWarn != 0)) {
            notes <- c(notes, sprintf(
              "alpha is at its %s search bound (alphaWarn = %d), so neither it nor the standard errors beside it are estimates",
              if (fit$alphaWarn == 1) "lower" else "upper", fit$alphaWarn))
          }
        }
        if (!is.null(fit$convergence)) {
          row$glm_converged <- as.integer(fit$convergence)
          if (isTRUE(fit$convergence != 0)) {
            notes <- c(notes, sprintf("phyloglm did not converge (code %d)",
                                      fit$convergence))
          }
        }
      }
    }
  }

  row$note <- paste(notes, collapse = "; ")
  rows[[length(rows) + 1]] <- as.data.frame(row, stringsAsFactors = FALSE)
}

out <- if (length(rows)) do.call(rbind, rows) else
  data.frame(scope = character(0), block = character(0))
write.table(out, out_file, sep = "\t", quote = FALSE, row.names = FALSE,
            na = "")
message(sprintf("wrote %d row(s) to %s", nrow(out), out_file))

# R's warnings are the only place some of these estimators say anything about
# how well they fitted, and "There were 16 warnings" in a log nobody expands is
# the same as silence.
w <- warnings()
if (length(w)) {
  message(sprintf("--- %d R warning(s) ---", length(w)))
  counts <- table(names(w))
  for (nm in names(sort(counts, decreasing = TRUE))) {
    message(sprintf("  [x%d] %s", counts[[nm]], nm))
  }
}
