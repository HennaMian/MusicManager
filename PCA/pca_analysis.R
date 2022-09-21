#!/usr/bin/env Rscript

library(corrplot)
library(factoextra)

music_matrix <- read.csv("music_matrix.csv", header=FALSE)

# run PCA; center and scale data
res.pca <- prcomp(music_matrix, center = TRUE, scale = TRUE)

summary(res.pca)
# Scree plot
pdf(file = "scree_plot.pdf", height = 10, width = 20)
fviz_eig(res.pca, addlabels = TRUE, ncp = 64)
dev.off()

# PCA plot; cos2 represents how important the PC for that sample
pdf(file = "pca_plot.pdf", height = 10, width = 10)
fviz_pca_ind(res.pca,
             col.ind = "cos2", # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)
dev.off()

# variance percent
eig.val <- get_eigenvalue(res.pca)
write.table(eig.val, file = "eigenvals_perc_var.txt")

# Assess contributions of each feature per dim
var <- get_pca_var(res.pca)
col1 <- colorRampPalette(c("#7F0000", "red", "#FF7F00", "yellow", "#7FFF7F",
                           "cyan", "#007FFF", "blue", "#00007F"))

pdf(file = "contribution_plot.pdf", height = 10, width = 10)
corrplot(var$contrib, is.corr=FALSE, col = col1(100))      
dev.off()

write.table(var$contrib, "feature_contributions.txt")


