import { Navigate } from "react-router-dom";

export default function ProtectedRoute({ children, requiredRole }) {
    const token = localStorage.getItem("officer_token");

    if (!token) {
        return <Navigate to="/kavindu/officer/login" replace />;
    }

    if (requiredRole) {
        const user = JSON.parse(localStorage.getItem("officer_user") || "{}");
        if (user.role !== requiredRole) {
            return <Navigate to="/kavindu/officer/dashboard" replace />;
        }
    }

    return children;
}
