package io.github.benfradet

import java.time.LocalDate

object Util {
  /** Compute the different dates between two dates */
  def daysBetween(fromDate: LocalDate, toDate: LocalDate): Seq[LocalDate] =
    fromDate.toEpochDay.to(toDate.toEpochDay).map(LocalDate.ofEpochDay)
}
