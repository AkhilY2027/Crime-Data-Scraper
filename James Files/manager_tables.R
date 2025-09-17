# Notes:
# - This does not handle overlapping tenures at different stores for the same manager. These
#   are listed as separate tenures.
#
# - Overlapping tenures, which are somewhat common are not listed as previous managers. For example,
#   if manager A is working from 2000-2025, and manager B works for a week at the same store in 
#   2022, manager B's time at the store will not have manager A listed as a previous manager
#
# Usage example at the bottom
#
# - example store to look at, company == "A" & unit_id == "STR00007"

get_manager_tenures <- function(rws_emp_staff_grp, cutoff_date, gap_days = 30) {
    # filter for job titles
    rws_manager_status <- rws_emp_staff_grp[((job_title_cd %ilike% "general" & job_title_cd %ilike% "manager") | (job_title_cd %ilike% "store" & job_title_cd %ilike% "manager")) & !job_title_cd %ilike% "assistant" & !job_title_cd %ilike% "training"]

    manager_tenures <- rws_manager_status[, .(company, unit_id, person_id, eff_date, end_date, job_title_cd)]
    cutoff_date <- as.IDate(cutoff_date)

    # Sort by person, company, unit, and start date
    setorder(manager_tenures, company, unit_id, person_id, eff_date)
    
    # Check gap between previous end_date and current eff_date
    manager_tenures[, gap_exceeded := {
        prev_end <- shift(end_date, type = "lag")
        gap_days_actual <- eff_date - prev_end
        c(TRUE, gap_days_actual[-1] > gap_days)  # First row always starts new group
    }, by = .(company, unit_id, person_id)]
    
    # Create group IDs
    manager_tenures[, group_id := cumsum(gap_exceeded), by = .(company, unit_id, person_id)]
    
    # Aggregate within each group - earliest start, latest end
    coalesced_tenures <- manager_tenures[, .(
        eff_date = min(eff_date),
        end_date = max(end_date),
        job_title_cd = paste(unique(job_title_cd), collapse = "; ")
    ), by = .(company, unit_id, person_id, group_id)]
    
    # Remove group_id and add tenure calculations
    coalesced_tenures[, group_id := NULL]
    coalesced_tenures[, current_tenure := fifelse(end_date >= cutoff_date, 1, 0)]
    coalesced_tenures[, tenure_length := fifelse(end_date >= cutoff_date, cutoff_date - eff_date, end_date - eff_date)]

    return(coalesced_tenures)
}

get_management_changes <- function(coalesced_tenures) {
    
    # Sort by store and start date
    setorder(coalesced_tenures, company, unit_id, eff_date)
    
    # Create result table
    result <- data.table()
    
    # Loop through each manager tenure
    for (i in 1:nrow(coalesced_tenures)) {
        current_row <- coalesced_tenures[i]
        
        # Find previous managers at this store who left before current manager started
        prev_managers <- coalesced_tenures[
            company == current_row$company & 
            unit_id == current_row$unit_id & 
            end_date <= current_row$eff_date & 
            person_id != current_row$person_id
        ]
        
        if (nrow(prev_managers) > 0) {
            # Get the one who left most recently (latest end_date), not longest tenure
            prev_manager <- prev_managers[which.max(end_date)]
            previous_manager_id <- prev_manager$person_id
            previous_manager_tenure <- prev_manager$tenure_length
        } else {
            previous_manager_id <- as.integer64(NA)
            previous_manager_tenure <- NA_integer_
        }
        
        # Find new manager's previous store
        new_mgr_prev <- coalesced_tenures[
            company == current_row$company & 
            person_id == current_row$person_id & 
            end_date < current_row$eff_date
        ]
        
        if (nrow(new_mgr_prev) > 0) {
            prev_store <- new_mgr_prev[which.max(end_date)]
            new_manager_previous_store <- prev_store$unit_id
            new_manager_previous_store_tenure <- prev_store$tenure_length
        } else {
            new_manager_previous_store <- NA_character_
            new_manager_previous_store_tenure <- NA_integer_
        }
        
        # Add row to result
        result <- rbind(result, data.table(
            company = current_row$company,
            new_manager_id = current_row$person_id,
            current_store = current_row$unit_id,
            start_date = current_row$eff_date,
            previous_manager_id = previous_manager_id,
            previous_manager_tenure = previous_manager_tenure,
            new_manager_previous_store = new_manager_previous_store,
            new_manager_previous_store_tenure = new_manager_previous_store_tenure
        ))
    }
    
    return(result)
}

manager_tenures <- get_manager_tenures(rws_emp_staff_grp, Sys.Date())

management_changes <- get_management_changes(manager_tenures)
