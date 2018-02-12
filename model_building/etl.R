source("./utils.R")
library(magrittr)
library(data.table)
library(caret)
library(mlbench)
library(tidyverse)
library(dtplyr)
library(doParallel)
library(onehot)

options(datatable.print.class = TRUE)
options(stringsAsFactors = FALSE)

# load data ---------------------------------------------------------------

flight_data <- fread("data/flight_data.csv")
flight_data_fact <- fread("data/flight_data_fact.csv")

flight_data %<>% big5_cols_to_utf8
flight_data_fact %<>% big5_cols_to_utf8

# etl ---------------------------------------------------------------------

dt_y <- flight_data[TAG_CD %in% c("10000011", "10000013"), 
                    .(CUST_ID, TAG_CD, TAG_VAL, TXN_CNT, YYYYMM)]

dt_x <- flight_data[!TAG_CD %in% c("10000011", "10000013"),
                    .(CUST_ID, TAG_CD, TAG_VAL, TXN_CNT, YYYYMM)]

# 辦卡問題
#>      TAG_CD            TAG_VAL
#>      <char>             <char>
#> 1: 10000010           辦商務卡
#> 2: 10000003 新辦卡_雙幣_昇恆昌
#> 3: 10000002             辦護照
dt_x[TAG_CD %in% c("10000003", "10000003"),
     YYYYMM := ifelse(YYYYMM < "201601", "201601", YYYYMM)]

# 辦護照問題 - 近3個月辦護照
dt_x[TAG_CD == "10000002",
     YYYYMM := ifelse(YYYYMM < "201601", "201601", YYYYMM)]

dt_x[TAG_CD == "10000003",
     TXN_CNT := if_else(as.integer(TXN_CNT) > 1, "1", TXN_CNT)]

# 過濾日期在 201601 之前的
dt_x <- dt_x[YYYYMM >= "201601" & YYYYMM < "201712"]

# Wide
dt_x_wide <- dt_x[, -c("TAG_VAL")] %>% 
  unique(by = c("YYYYMM", "CUST_ID", "TAG_CD")) %>% 
  dcast(YYYYMM +  CUST_ID ~ TAG_CD, value.var = "TXN_CNT", fill=0)

# var ---------------------------------------------------------------------

# Convert to integer
vars <- dt_x_wide[, names(.SD), .SDcols = `00000445`:`10000010`]
dt_x_wide[, (vars) := lapply(.SD, as.integer), .SDcols = vars]


# Seasonal
dt_season <- list(Q1 = c("01", "02"),
                  Q2 = c("03", "04", "05"),
                  Q3 = c("06", "07", "08"),
                  Q4 = c("09", "10", "11", "12")) %>% 
  melt %>% setDT %>% setnames(c("MM", "season"))

dt_x_wide[, MM := YYYYMM %>% substr(5, 6)]
dt_x_wide <- dt_x_wide %>% merge(dt_season, by = c("MM"))

dt_x_wide[, season := season %>% as.factor]

# Feature engineering part 1 -----------------------------------------------

summary(dt_x_wide)
# y1          y2        
# Min.   :0   Min.   : 1.000  
# 1st Qu.:0   1st Qu.: 1.000  
# Median :0   Median : 1.000  
# Mean   :0   Mean   : 1.777  
# 3rd Qu.:0   3rd Qu.: 2.000  
# Max.   :0   Max.   :35.000  


# dt_all_temp3[`10000005` > 8, .N] # N = 78
dt_x_wide <- dt_x_wide[`10000005` <= 8] # 訂房網


# 補完日期 (23 個月) ----------------------------------------------

# comb <- dt_x_wide %>% expand(CUST_ID, YYYYMM)
# comb %>%
#   group_by(CUST_ID) %>%
#   summarise(n = n_distinct(YYYYMM)) %>%
#   select(n) %>%
#   table

# dt_x_wide_tmp <- dt_x_wide %>% 
#   complete(CUST_ID, YYYYMM) %>% 
#   setDT()
# 
# cols = c("00000445", "00001184", "00001185", "10000001", "10000002", "10000003",
#          "10000004", "10000005", "10000007", "10000008", "10000010")
# dt_x_wide_tmp[, (cols) := lapply(.SD, fill_na, fill = "0"), 
#               .SDcols = cols]
# 
# dt_x_wide <- dt_x_wide_tmp

# 10000010 辦商務卡 binary -> 持有商務卡 (辦卡後即為1)
fill_0 <- function(x) {
  # x = c(0, 0, 0, 1, 0, 0, 0)
  # print(x)
  if (sum(x) == 0 || length(x) == 1) {
    return(x)
  }
  zero_position <- min(which(x == 1L))
  x[zero_position:length(x)] = 1L
  x
}
dt_x_wide %>% setkey(CUST_ID, YYYYMM)
dt_x_wide[, `10000010` := .(fill_0(`10000010`)), .(CUST_ID)] # 辦商務卡
dt_x_wide[, `10000003` := .(fill_0(`10000003`)), .(CUST_ID)] # 辦卡_雙幣_昇恆昌

# Fact table ---------------------------------------------------------------

flight_data_fact_wide <- flight_data_fact %>% 
  dcast(CUST_ID ~ TAG_NAME, value.var = "VAL") %>% setDT

flight_data_fact_wide %>% 
  setnames(
    old = c("年收入", "年齡", "性別", "近一年總刷卡金額_信用卡", "近四年信用卡消費國家數", "教育程度"),
    new = c("income", "age", "sex", "spending", "n_country", "edu")
  )

flight_data_fact_wide[, `:=`(
  income = income %>% as.numeric(),
  age = age %>% as.integer(),
  spending = spending %>% as.numeric(),
  n_country = n_country %>% as.integer()
)]

flight_data_fact_wide %<>% .[is.na(n_country), n_country := 0]
flight_data_fact_wide %<>% .[is.na(n_country), n_country := 0]
flight_data_fact_wide %<>% .[is.na(n_country), n_country := 0]

# NA imputation
flight_data_fact_wide %>% sapply(anyNA) # check NA
IMPUTE_VAR = c("spending", "income", "age")
preproc <- preProcess(flight_data_fact_wide[, IMPUTE_VAR, with = FALSE],
                      method = "medianImpute")
flight_data_fact_wide <- predict(preproc, flight_data_fact_wide)
flight_data_fact_wide[, .N, edu][N == max(N)] # "03.大學"

flight_data_fact_wide[is.na(edu), edu := "03.大學"]

# Data Engineering part 2 -------------------------------------------------

# clip n_country
flight_data_fact_wide[, n_country := rcppClamp(n_country, 0, 12)]

# log transform
flight_data_fact_wide[, income := log(income + 0.0001)]
flight_data_fact_wide[, income := income - median(income)]

flight_data_fact_wide[, spending := log(spending + 0.0001)]
flight_data_fact_wide[, spending := spending - median(spending)]

# Cut age
# flight_data_fact_wide[, age] %>% summary
flight_data_fact_wide <- flight_data_fact_wide[age >= 20 & age <= 80]
flight_data_fact_wide[, age_cut := cut(age, breaks = c(20, 27, 35, 45, 55, 65, Inf), 
                                       include.lowest = T)]

# Ordered
flight_data_fact_wide[, edu := 5-(edu %>% as.ordered %>% as.numeric)]

# Combine Xs and Ys -------------------------------------------------------

dt_all_temp <- dt_x_wide %>% 
  merge(flight_data_fact_wide, by = "CUST_ID")

dt_all_temp2 <- dt_all_temp %>% 
  merge(dt_y[TAG_CD == "10000011", .(CUST_ID, YYYYMM, y1 = TXN_CNT)],
        by = c("CUST_ID", "YYYYMM"), all.x = TRUE) %>% 
  merge(dt_y[TAG_CD == "10000013", .(CUST_ID, YYYYMM, y2 = TXN_CNT)],
        by = c("CUST_ID", "YYYYMM"), all.x = TRUE) %>% 
  .[, c("y1", "y2") := lapply(.SD, as.integer), .SDcols = c("y1", "y2")]


dt_all_temp2[, c("y1", "y2") := lapply(.SD, fill_na, fill = 0L), 
             .SDcols = c("y1", "y2")]

dt_all_temp2 %>% sapply(anyNA) %>% any # check NA
dt_all_temp2 %>% setkey(CUST_ID, YYYYMM)
dt_all_temp2[, y1 := NULL]

# Filter bad data ---------------------------------------------------------

dt_all_temp3 <- copy(dt_all_temp2)[, y2 := ifelse(y2 >= 1, 1L, 0L)] # 每月出國
dt_all_temp3[, y2 := as.factor(y2)]

dt_all_temp3[, y2] %>% summary

# onehot ------------------------------------------------------------------

onehot_var <- c("age_cut", "sex", "season")
encoder <- dt_all_temp3[, ..onehot_var] %>% onehot(stringsAsFactors=TRUE)
dt_all_temp4 <- cbind(dt_all_temp3[, -c("age_cut", "sex", "season")],
                      predict(encoder, dt_all_temp3[, ..onehot_var] ))


# Export data -------------------------------------------------------------

fwrite(dt_all_temp4, "./data/data_cleaned.csv")

