#
# Conditions/Assumptions:
# 1. The primary condition that is filtered out is the 0.1% of employees' work history
#    that has multiple different date_of_hire listed, as it is corrupted most likely.
#
# Two functions:
# 1. process_employement_data() returns the monthly data
# 2. process_employment_stints() returns the raw stints
#
# example employees to check:
# - company == "A" & person_id == 130154
# - company == "A" & person_id == 53929
# Usage examples:
# Default behavior:
# stint_data <- process_employment_stints(rws_emp_status, coalesce_days = 30)

# With end cutoff:
# stint_data <- process_employment_stints(rws_emp_status, coalesce_days = 30, 
#                                        end_cutoff_date = "2024-12-31")
# Function to process employment data for all persons
process_employment_data <- function(dt, coalesce_days = 1, end_cutoff_date = NULL, start_cutoff_date = NULL) {
  
  # Set default cutoff dates if not provided and ensure IDate type
  if (is.null(end_cutoff_date)) {
    end_cutoff_date <- as.IDate(Sys.Date())
  } else {
    end_cutoff_date <- as.IDate(end_cutoff_date)
  }
  
  if (!is.null(start_cutoff_date)) {
    start_cutoff_date <- as.IDate(start_cutoff_date)
  }
 
# Step 1: Coalesce overlapping/close stints
coalesce_stints <- function(stint_dt, days_threshold = coalesce_days) {
  if (nrow(stint_dt) <= 1) return(stint_dt)
  
  result <- stint_dt[1]
  
  for (i in 2:nrow(stint_dt)) {
    current <- stint_dt[i]
    last_idx <- nrow(result)
    last_row <- result[last_idx]
    
    # Handle NA end dates for comparison - use end_cutoff_date for ongoing stints
    last_end <- if (is.na(last_row$stint_end)) end_cutoff_date else last_row$stint_end
    
    # Check if current stint overlaps or is within threshold days
    overlap_condition <- current$stint_start <= last_end
    close_condition <- as.numeric(current$stint_start - last_end) <= days_threshold
    
    if (overlap_condition || close_condition) {
      # Coalesce: extend the end date (keeping NA if either is NA and it's the later one)
      if (is.na(current$stint_end)) {
        result[last_idx, stint_end := NA] 
      } else if (!is.na(last_row$stint_end)) {
        result[last_idx, stint_end := pmax(last_row$stint_end, current$stint_end)]
      }
    } else {
      # No overlap/close enough: add as new stint
      result <- rbind(result, current)
    }
  }
  
  return(result)
}

# Step 2: Expand stints to monthly records
expand_to_monthly <- function(stint_dt) {
  stint_dt[, {
    # Determine if stint is truly ongoing (NA term_date) vs ends after cutoff
    is_truly_ongoing <- is.na(stint_end)
    stint_ends_after_cutoff <- !is.na(stint_end) & stint_end > end_cutoff_date
    
    # Calculate actual end date for month calculation
    actual_end <- fifelse(is_truly_ongoing, 
                         end_cutoff_date, 
                         pmin(stint_end, end_cutoff_date))
    
    # For churn flag purposes: ongoing if truly ongoing OR if ends after cutoff
    is_ongoing_for_churn <- is_truly_ongoing | stint_ends_after_cutoff
    
    # Calculate month sequences
    start_ym <- year(stint_start) * 12 + month(stint_start) - 1
    end_ym <- year(actual_end) * 12 + month(actual_end) - 1
    months_count <- end_ym - start_ym + 1
    
    # Handle each stint individually
    lapply(seq_along(months_count), function(idx) {
      mc <- months_count[idx]
      if (mc <= 0) {
        data.table(year = integer(0), month = integer(0), churn_flag = integer(0))
      } else {
        month_indices <- 0:(mc - 1)
        years <- (start_ym[idx] + month_indices) %/% 12
        months <- ((start_ym[idx] + month_indices) %% 12) + 1
        
        churn_flags <- rep(0L, mc)
        # Only set churn flag if the stint actually ended before or at the cutoff
        # AND it's not considered ongoing for churn purposes
        if (!is_ongoing_for_churn[idx] && mc > 0) {
          churn_flags[mc] <- 1L
        }
        
        data.table(year = years, month = months, churn_flag = churn_flags)
      }
    }) |> rbindlist()
  }, by = .(company, person_id)]
}

dt <- dt[dt[, uniqueN(date_of_hire), by=.(company, person_id)][V1 == 1], on = .(company, person_id)]
 
 # Main processing 
 result <- dt[, {
   # Create stint start and end dates - ensure IDate types
   stint_start <- fifelse(is.na(date_of_rehire), date_of_hire, date_of_rehire)
   stint_end <- term_date  # This should already be IDate or NA
   
   # Create stints data.table and sort
   stints <- data.table(company = company, person_id = person_id, 
                       stint_start = as.IDate(stint_start), 
                       stint_end = as.IDate(stint_end))
   stints <- stints[order(stint_start)]
   
   # Coalesce overlapping/close stints
   coalesced <- coalesce_stints(stints, coalesce_days)
   
   # Expand to monthly records
   monthly <- expand_to_monthly(coalesced)
   
   monthly
 }, by = .(company, person_id)]
 
 # Clean up and sort
 result <- result[, .(company, person_id, year, month, churn_flag)]
 result <- result[order(company, person_id, year, month)]
 
 # Apply start cutoff filter at the very end to avoid corrupting data
 if (!is.null(start_cutoff_date)) {
   # Convert year/month back to date for comparison (using first day of month)
   result[, month_date := as.IDate(paste(year, month, "01", sep = "-"))]
   result <- result[month_date >= start_cutoff_date]
   result[, month_date := NULL]
 }
 
 return(result)
}

# Usage examples:
# monthly_employment <- process_employment_data(rws_emp_status, coalesce_days = 30)

# Function to process employment data and return coalesced stints only
process_employment_stints <- function(dt, coalesce_days = 1, end_cutoff_date = NULL) {
  
  # Set default cutoff date if not provided and ensure IDate type
  if (is.null(end_cutoff_date)) {
    end_cutoff_date <- as.IDate(Sys.Date())
  } else {
    end_cutoff_date <- as.IDate(end_cutoff_date)
  }
  
  # Step 1: Coalesce overlapping/close stints
  coalesce_stints <- function(stint_dt, days_threshold = coalesce_days) {
    if (nrow(stint_dt) <= 1) return(stint_dt)
    
    result <- stint_dt[1]
    
    for (i in 2:nrow(stint_dt)) {
      current <- stint_dt[i]
      last_idx <- nrow(result)
      last_row <- result[last_idx]
      
      # Handle NA end dates for comparison - use end_cutoff_date for ongoing stints
      last_end <- if (is.na(last_row$stint_end)) end_cutoff_date else last_row$stint_end
      
      # Check if current stint overlaps or is within threshold days
      overlap_condition <- current$stint_start <= last_end
      close_condition <- as.numeric(current$stint_start - last_end) <= days_threshold
      
      if (overlap_condition || close_condition) {
        # Coalesce: extend the end date (keeping NA if either is NA and it's the later one)
        if (is.na(current$stint_end)) {
          result[last_idx, stint_end := NA]  # Keep as IDate NA
        } else if (!is.na(last_row$stint_end)) {
          result[last_idx, stint_end := pmax(last_row$stint_end, current$stint_end)]
        }
      } else {
        # No overlap/close enough: add as new stint
        result <- rbind(result, current)
      }
    }
    
    return(result)
  }

  dt <- dt[dt[, uniqueN(date_of_hire), by=.(company, person_id)][V1 == 1], on = .(company, person_id)]
  
  # Main processing pipeline - create and coalesce stints only
  result <- dt[, {
    # Create stint start and end dates - ensure IDate types
    stint_start <- fifelse(is.na(date_of_rehire), date_of_hire, date_of_rehire)
    stint_end <- term_date  # This should already be IDate or NA
    
    # Create stints data.table and sort
    stints <- data.table(company = company, person_id = person_id, 
                        stint_start = as.IDate(stint_start), 
                        stint_end = as.IDate(stint_end))
    stints <- stints[order(stint_start)]
    
    # Coalesce overlapping/close stints
    coalesced <- coalesce_stints(stints, coalesce_days)
    
    coalesced
  }, by = .(company, person_id)]
  
  # Clean up and sort final result
  result <- result[order(company, person_id, stint_start)]
  
  return(result)
}

#