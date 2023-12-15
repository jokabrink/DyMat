model Model
  Real x(start=1, fixed=true) "välue";
  Real v = der(x) "velocity ⚡";
  parameter Real alpha=1 "time constant";
equation 
  0 = v + alpha*x;

  annotation (
    Icon(coordinateSystem(preserveAspectRatio=false)),
    Diagram(coordinateSystem(preserveAspectRatio=false)),
    experiment(
      StopTime=10,
      Interval=1));
end Model;
