import "./Home.css";

function Home() {
  return (
    <div className="home">
      <h1 className="home__title">
        AI-Smart Animal Vehicle <br /> Collision Predictor
      </h1>

      <p className="home__subtitle">
        Sri Lanka records frequent wildlife–vehicle collisions, especially involving elephants.
        This platform uses Artificial Intelligence and geospatial analysis to identify high-risk
        locations and support wildlife conservation and road safety planning.
      </p>

      <div className="home__stats">
        <StatCard icon="🐘" number="7,000+" text="Estimated elephant population in Sri Lanka" />
        <StatCard icon="🚧" number="260+" text="Wildlife–vehicle collision locations analyzed" />
        <StatCard icon="🌿" number="Wildlife Dept." text="Data supported by conservation authorities" />
      </div>

      <div className="home__featureWrap">
        <div className="home__cards">
          <FeatureCard
            title="📍 Risk Mapping"
            text="Visualizes high-risk road/rail segments using historical collision data, spatial clustering, and geospatial analysis."
          />
          <FeatureCard
            title="🤖 AI-Based Analysis"
            text="Machine learning analyzes environmental factors (NDVI, rainfall, distance to forests) to estimate collision risk."
          />
          <FeatureCard
            title="🐘 Conservation Support"
            text="Helps wildlife officers and planners reduce human–elephant conflict and improve safety decisions."
          />
        </div>
      </div>

      <div className="home__explore">
        <span>Explore More</span>
        <div className="home__arrow">↓</div>
      </div>
    </div>
  );
}

function StatCard({ icon, number, text }) {
  return (
    <div className="stat">
      <div className="stat__icon">{icon}</div>
      <h3 className="stat__number">{number}</h3>
      <p className="stat__text">{text}</p>
    </div>
  );
}

function FeatureCard({ title, text }) {
  return (
    <div className="card">
      <h3 className="card__title">{title}</h3>
      <p className="card__text">{text}</p>
    </div>
  );
}

export default Home;