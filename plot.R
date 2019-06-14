library(gridExtra)
library(ggplot2)
library(dplyr)

scatter_rsa <- function(){
    # Figure fig:scatter_rsa_prefix
    data <- read.csv("report/scatter_rsa.csv")

    plot0 <- ggplot(data, aes(x=tree, y=rep0)) + geom_point(alpha=0.02, shape = '.') + 
        geom_smooth(method='lm') + 
        xlab("Tree Kernel") +
        ylab("Random encoder") + 
        theme(text=element_text(size=16))

    plot1 <- ggplot(data, aes(x=tree, y=rep1)) + geom_point(alpha=0.02, shape = '.') + 
        geom_smooth(method='lm') + 
        xlab("Tree Kernel") +
        ylab("Trained encoder") +
        theme(text=element_text(size=16))

    #grid.arrange(plot0, plot1, nrow=2) 
    g <- arrangeGrob(plot0, plot1, nrow=2) 
    ggsave(file="report/scatter_rsa.png", g)
}


size_distribution <- function() {
    # Figure fig:size_dist

    data <- read.csv("report/size_distribution.csv")

    plot1 <- ggplot(data, aes(x=`d_2.0`)) + geom_histogram(binwidth=1) + 
        xlab("size") + ylab(NULL) +
        xlim(1,16) + ylim(0, 3000) +
        labs(title="d=2.0") +
        theme(text=element_text(size=16), axis.text.y=element_blank(), axis.ticks.y=element_blank())
    plot2 <- ggplot(data, aes(x=`d_1.8`)) + geom_histogram(binwidth=1) +
        xlab("size") + ylab(NULL) +
        xlim(1,16) + ylim(0, 3000) +
        labs(title="d=1.8") +
        theme(text=element_text(size=16), axis.text.y=element_blank(), axis.ticks.y=element_blank())
    plot3 <- ggplot(data, aes(x=`d_1.5`)) + geom_histogram(binwidth=1) +
        xlab("size") + ylab(NULL) +
        labs(title="d=1.5") +
        xlim(1,16) + ylim(0, 3000) +
        theme(text=element_text(size=16), axis.text.y=element_blank(), axis.ticks.y=element_blank())
    g <- arrangeGrob(plot1, plot2, plot3, nrow=3) 
    ggsave(file="report/size_distribution.png", g)

}

BERT_layers <- function(size='') {
    # Figure fig:bert-layers
    data <- read.table("report/BERT_layers.csv", header=TRUE) 
    data  %>% mutate(α=factor(alpha)) %>% filter(encoder==paste('BERT',size, sep='') & step == 'first' & α != 0.75) %>% 
        mutate(metric=recode(metric, `RSA_regress`='RSA regress')) %>%
        ggplot(aes(x=layer, y=r, color=α, linetype=α, shape=mode)) + 
        geom_line() + geom_point() +
        facet_wrap(~ metric, nrow=2, scales="free_y") +
        xlab(NULL) + 
        theme(text=element_text(size=16), legend.position = "bottom")

    ggsave(file=paste('report/BERT',size, "_layers.png", sep=''),width = 5*3, height = 5*3, units = "cm")
}

sim_distribution <- function() {
    # Figure fig:sim_distribution
    data <- read.table("report/sim_distribution.txt", header=TRUE)
    data  %>% filter(α != 0.75) %>% ggplot(aes(group=α, x=α, y=K)) + geom_jitter(width = 0.2, alpha=0.1) + 
    ylab(expression("K'")) +
    xlab("λ") + 
    scale_x_continuous(breaks=c(0.5, 1.0)) +
    coord_flip()
    theme(text=element_text(size=16))
    ggsave("report/sim_distribution.png", width = 4*3, height = 2*3, units = "cm")
}
#scatter_rsa();
#size_distribution();
#BERT_layers('24');
#sim_distribution();

