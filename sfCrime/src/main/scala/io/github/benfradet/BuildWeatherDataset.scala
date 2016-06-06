package io.github.benfradet

import java.io.FileWriter
import java.time.LocalDate
import java.time.format.DateTimeFormatter

import com.twitter.util.{JavaTimer, TimerTask, Duration}
import io.circe.generic.auto._
import io.circe.syntax._
import org.slf4j.LoggerFactory

import scala.collection.mutable
import scala.io.Source
import scala.util.{Failure, Success, Try}

sealed trait WeatherDataTrait {
  def temperatureC: Double
  def weather: String
}
case class WeatherData(temperatureC: Double, weather: String) extends WeatherDataTrait
object WeatherData {
  def parseWeatherData(unparsed: String): WeatherData = {
    val fields = unparsed.split(",")
    // the temp is the second field and the weather condition the 12th
    WeatherData(fields(1).toDouble, fields(11))
  }
}
case class WeatherDataDate(date: String, temperatureC: Double, weather: String)
  extends WeatherDataTrait
object WeatherDataDate {
  /**
   * Build a weather data date object with the avg temperature and the most occurring weather
   * condition
   */
  def buildWeatherDataDate(date: String, wds: Seq[WeatherData]): WeatherDataDate =
    WeatherDataDate(
      date,
      wds.map(_.temperatureC).sum / wds.size,
      wds
        .map(_.weather)
        .foldLeft(Map.empty[String, Int]) { (acc, word) =>
          acc + (word -> (acc.getOrElse(word, 0) + 1))
        }
        .toSeq.sortBy(-_._2).head._1
    )
}

/**
 * Build a dataset containing weather data for the different dates contained in the SF crime
 * dataset.
 */
object BuildWeatherDataset {
  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val minDate = LocalDate.of(2003, 1, 1)
    val maxDate = LocalDate.of(2015, 5, 14)
    val dateFormatter = DateTimeFormatter.ISO_LOCAL_DATE

    val urlTemplate =
      "https://www.wunderground.com/history/airport/KSFO/%d/%d/%d/DailyHistory.html?format=1"
    val urlsQeueue = mutable.Queue(
      Util.daysBetween(minDate, maxDate)
        .map { d =>
          val formattedDate = d.format(dateFormatter)
          (formattedDate, urlTemplate.format(d.getYear, d.getMonthValue, d.getDayOfMonth))
        }: _*
    )
    require(urlsQeueue.nonEmpty, "there should be URLs to query")

    val fileWriter = new FileWriter("sfCrime/src/main/resources/weather.json")
    fileWriter.write("[\n")
    val timer = new JavaTimer(isDaemon = false)
    lazy val task: TimerTask = timer.schedule(Duration.fromSeconds(5)) {
      val (date, url) = urlsQeueue.dequeue()
      Try(Source.fromURL(url).getLines().drop(2)) match {
        case Success(lines) =>
          val wds = lines.map(WeatherData.parseWeatherData).toSeq
          val wdd = WeatherDataDate.buildWeatherDataDate(date, wds)
          val json = wdd.asJson.spaces2
          if (urlsQeueue.isEmpty) {
            task.cancel()
            timer.stop()
            fileWriter.write(s"$json\n")
            fileWriter.write("]")
            fileWriter.close()
          } else {
            fileWriter.write(s"$json,\n")
          }
        case Failure(ex) => logger.warn("couldn't retrieve weather data", ex)
      }
    }
    require(task != null)
  }
}
