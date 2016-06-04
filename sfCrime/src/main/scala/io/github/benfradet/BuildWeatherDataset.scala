package io.github.benfradet

import java.io.FileWriter
import java.time.LocalDate
import java.time.format.DateTimeFormatter

import com.twitter.util.{JavaTimer, TimerTask, Duration}
import io.circe.generic.auto._
import io.circe.syntax._
import org.slf4j.LoggerFactory
import shapeless._
import syntax.std.traversable._
import syntax.std.tuple._

import scala.collection.mutable
import scala.io.Source
import scala.util.{Failure, Success, Try}

sealed trait WeatherDataTrait {
  def temperatureC: Double
  def weather: String
}
case class WeatherData(temperatureC: Double, weather: String) extends WeatherDataTrait
case class WeatherDataDate(date: String, temperatureC: Double, weather: String)
  extends WeatherDataTrait

/**
 * Build a dataset containing weather data for the different dates contained in the SF crime
 * dataset.
 */
object BuildWeatherDataset {
  val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    val minDate = LocalDate.of(2003, 1, 1)
    val maxDate = LocalDate.of(2015, 1, 14)
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

    // indexes of the relevant fields: 1 temperature and 11 weather condition
    val idxs = Array(1, 11)

    val fileWriter = new FileWriter("sfCrime/src/main/resources/weather.json")
    fileWriter.write("[\n")
    val timer = new JavaTimer(isDaemon = false)
    lazy val task: TimerTask = timer.schedule(Duration.fromSeconds(5)) {
      val (date, url) = urlsQeueue.dequeue()
      Try(Source.fromURL(url).getLines().drop(2)) match {
        case Success(lines) =>
          val wds = lines.map(parseWeatherData(_, idxs)).toSeq
          val wdd = buildWeatherDataDate(date, wds)
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

  def parseWeatherData(unparsed: String, idxsToKeep: Array[Int]): WeatherData = {
    val tuple = unparsed.split(",")
      .zipWithIndex
      .filter(idxsToKeep contains _._2)
      .map(_._1)
      .toHList[String :: String :: HNil]
      .get.tupled
    WeatherData(tuple._1.toDouble, tuple._2)
  }

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
