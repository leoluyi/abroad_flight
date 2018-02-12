library(Rcpp)

cppFunction('NumericVector rcppClamp(NumericVector x, double mi, double ma) {
               return clamp(mi, x, ma);
            }')

check_enc <- function(df) {
  sapply(df,
         function(col) {
           ifelse(class(col) == "character", 
                  Encoding(col), NA)
         })
}

utf8_cols_to_big5 <- function(df) {
  # df = CUBE_PERSONNEL_QUOTA_EXCEL
  is_utf8_cols <- sapply(df,
                         function(col) {
                           ifelse(class(col) == "character", 
                                  any(Encoding(col) == "UTF-8")
                                  , FALSE)})
  utf8_cols <- names(df)[is_utf8_cols]
  for (col in utf8_cols) {
    df[[col]] <- iconv(df[[col]], "UTF-8", "Big5")
  }
  df
}


big5_cols_to_utf8 <- function(df) {
  # df = CUBE_PERSONNEL_QUOTA_EXCEL
  is_utf8_cols <- sapply(df,
                         function(col) {
                           ifelse(class(col) == "character", 
                                  any(Encoding(col) == "unknown")
                                  , FALSE)})
  utf8_cols <- names(df)[is_utf8_cols]
  for (col in utf8_cols) {
    df[[col]] <- iconv(df[[col]], "CP950", "UTF-8")
  }
  df
}


fill_na <- function(x, fill) {
  ifelse(is.na(x), fill, x)
}
