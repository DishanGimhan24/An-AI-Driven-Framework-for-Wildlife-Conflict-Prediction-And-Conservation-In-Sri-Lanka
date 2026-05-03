import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  StyleSheet,
  Linking,
  StatusBar,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';

// Mirrors the easeOutQuart animated counter from the web Home.jsx
function AnimatedCounter({ end, duration = 2000, suffix = '' }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime = null;
    let frameId;

    const step = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const progress = Math.min((timestamp - startTime) / duration, 1);
      const easeProgress = 1 - Math.pow(1 - progress, 4);
      setCount(Math.floor(easeProgress * end));
      if (progress < 1) {
        frameId = requestAnimationFrame(step);
      }
    };

    frameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameId);
  }, [end, duration]);

  return (
    <Text style={styles.counterValue}>
      {count.toLocaleString()}
      {suffix}
    </Text>
  );
}

export default function HomeScreen({ navigation }) {
  const insets = useSafeAreaInsets();
  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <StatusBar barStyle="light-content" backgroundColor="#0f172a" />
      <ScrollView
        style={styles.container}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* ── Hero ─────────────────────────────────────────────── */}
        <View style={styles.hero}>
          <Text style={styles.heroTitle}>🐘 Wildlife Safety System</Text>
          <Text style={styles.heroSubtitle}>
            Community reporting for citizens and an operational dashboard for
            wildlife officers. Protecting Sri Lanka's precious wildlife through
            AI-powered early warning systems.
          </Text>
        </View>

        {/* ── Live Impact Counters ──────────────────────────────── */}
        <View style={styles.countersSection}>
          <View style={[styles.glassCard, styles.counterCard, { borderTopColor: '#818cf8' }]}>
            <Feather name="trending-up" size={28} color="#818cf8" style={styles.counterIcon} />
            <AnimatedCounter end={1240} suffix="+" />
            <Text style={styles.counterLabel}>REPORTS SUBMITTED</Text>
          </View>

          <View style={[styles.glassCard, styles.counterCard, { borderTopColor: '#f87171' }]}>
            <MaterialCommunityIcons name="shield-alert-outline" size={28} color="#f87171" style={styles.counterIcon} />
            <AnimatedCounter end={85} />
            <Text style={styles.counterLabel}>ARRESTS ASSISTED BY AI</Text>
          </View>

          <View style={[styles.glassCard, styles.counterCard, { borderTopColor: '#4ade80' }]}>
            <MaterialCommunityIcons name="leaf" size={28} color="#4ade80" style={styles.counterIcon} />
            <AnimatedCounter end={300} suffix="+" />
            <Text style={styles.counterLabel}>ANIMALS SAVED</Text>
          </View>
        </View>

        {/* ── How It Works ─────────────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>How It Works</Text>

          <View style={styles.glassCard}>
            <View style={[styles.stepIconWrap, { backgroundColor: 'rgba(99,102,241,0.15)' }]}>
              <Feather name="eye" size={32} color="#818cf8" />
            </View>
            <Text style={styles.stepTitle}>1. You Spot It</Text>
            <Text style={styles.stepText}>
              Observe suspected poaching, illegal logging, traps, or distressed
              wildlife happening near you.
            </Text>
          </View>

          <View style={styles.glassCard}>
            <View style={[styles.stepIconWrap, { backgroundColor: 'rgba(239,68,68,0.15)' }]}>
              <Feather name="smartphone" size={32} color="#f87171" />
            </View>
            <Text style={styles.stepTitle}>2. You Report It</Text>
            <Text style={styles.stepText}>
              Quickly submit a confidential report with precise GPS coordinates
              and photo evidence right from your phone.
            </Text>
          </View>

          <View style={styles.glassCard}>
            <View style={[styles.stepIconWrap, { backgroundColor: 'rgba(34,197,94,0.15)' }]}>
              <MaterialCommunityIcons name="shield-check" size={32} color="#4ade80" />
            </View>
            <Text style={styles.stepTitle}>3. We Intervene</Text>
            <Text style={styles.stepText}>
              The system immediately alerts and dispatches the closest Rapid
              Response Team to handle the threat.
            </Text>
          </View>
        </View>

        {/* ── Know What To Report ───────────────────────────────── */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Know What To Report</Text>
          <Text style={styles.sectionSubtitle}>
            Unsure if what you saw is illegal? Here are the most common offences
            our department handles.
          </Text>

          <View
            style={[
              styles.glassCard,
              styles.reportTypeCard,
              { borderLeftColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.05)' },
            ]}
          >
            <View style={[styles.reportTypeIcon, { backgroundColor: 'rgba(239,68,68,0.2)' }]}>
              <Feather name="scissors" size={24} color="#fca5a5" />
            </View>
            <View style={styles.reportTypeText}>
              <Text style={styles.reportTypeTitle}>Wire Snares &amp; Traps</Text>
              <Text style={styles.reportTypeDesc}>
                Hidden wire loops used to catch wildlife. Often found tied to trees
                near watering holes or trails.
              </Text>
            </View>
          </View>

          <View
            style={[
              styles.glassCard,
              styles.reportTypeCard,
              { borderLeftColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.05)' },
            ]}
          >
            <View style={[styles.reportTypeIcon, { backgroundColor: 'rgba(245,158,11,0.2)' }]}>
              <MaterialCommunityIcons name="axe" size={24} color="#fcd34d" />
            </View>
            <View style={styles.reportTypeText}>
              <Text style={styles.reportTypeTitle}>Illegal Logging</Text>
              <Text style={styles.reportTypeDesc}>
                Unmarked trucks carrying valuable timber (like Rosewood) or sounds
                of chainsaws deep inside protected reserves.
              </Text>
            </View>
          </View>

          <View
            style={[
              styles.glassCard,
              styles.reportTypeCard,
              { borderLeftColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.05)' },
            ]}
          >
            <View style={[styles.reportTypeIcon, { backgroundColor: 'rgba(59,130,246,0.2)' }]}>
              <Feather name="alert-triangle" size={24} color="#93c5fd" />
            </View>
            <View style={styles.reportTypeText}>
              <Text style={styles.reportTypeTitle}>Distressed Wildlife</Text>
              <Text style={styles.reportTypeDesc}>
                Animals that are visibly injured, wandering into human settlements,
                or baby animals that appear orphaned.
              </Text>
            </View>
          </View>
        </View>

        {/* ── Community Reporting Card ──────────────────────────── */}
        <View style={[styles.glassCard, styles.cta]}>
          <Text style={styles.ctaIcon}>📝</Text>
          <Text style={styles.ctaTitle}>Community Reporting</Text>
          <Text style={styles.ctaDesc}>
            Report illegal poaching or wildlife offences. Quick and simple. Your
            report helps protect wildlife.
          </Text>

          <TouchableOpacity
            style={styles.reportBtn}
            onPress={() => navigation.navigate('Report')}
            activeOpacity={0.8}
          >
            <Text style={styles.reportBtnText}>Make a Report</Text>
          </TouchableOpacity>

          <View style={styles.anonymousBadge}>
            <MaterialCommunityIcons name="shield-check" size={16} color="#4ade80" />
            <Text style={styles.anonymousText}>100% Anonymous Guarantee</Text>
          </View>
        </View>

        {/* ── Emergency Hotline ─────────────────────────────────── */}
        <View style={styles.emergencyCard}>
          <Feather name="phone-call" size={32} color="#f87171" style={{ marginBottom: 12 }} />
          <Text style={styles.emergencyTitle}>Wildlife Emergency?</Text>
          <Text style={styles.emergencyDesc}>
            If human life is in immediate danger from wildlife, do not use the
            reporting form. Call the department directly.
          </Text>
          <TouchableOpacity
            style={styles.callBtn}
            onPress={() => Linking.openURL('tel:1992')}
            activeOpacity={0.8}
          >
            <Text style={styles.callBtnText}>Call 1992 Hotline</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 48,
  },

  // Hero
  hero: {
    backgroundColor: 'rgba(99,102,241,0.1)',
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(99,102,241,0.2)',
    alignItems: 'center',
  },
  heroTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 12,
  },
  heroSubtitle: {
    fontSize: 14,
    color: '#94a3b8',
    textAlign: 'center',
    lineHeight: 22,
  },

  // Counters
  countersSection: {
    flexDirection: 'column',
    gap: 12,
    marginBottom: 24,
  },
  glassCard: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
    padding: 16,
    marginBottom: 12,
  },
  counterCard: {
    flexDirection: 'row',
    alignItems: 'center',
    borderTopWidth: 4,
    paddingVertical: 18,
    gap: 16,
    marginBottom: 0,
  },
  counterIcon: {},
  counterValue: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    flex: 1,
  },
  counterLabel: {
    color: '#94a3b8',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    flexShrink: 1,
  },

  // Sections
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 16,
  },
  sectionSubtitle: {
    fontSize: 14,
    color: '#94a3b8',
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 20,
  },

  // How it works steps
  stepIconWrap: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
    alignSelf: 'center',
  },
  stepTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 8,
  },
  stepText: {
    fontSize: 13,
    color: '#94a3b8',
    textAlign: 'center',
    lineHeight: 20,
  },

  // Know what to report cards
  reportTypeCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    borderLeftWidth: 4,
    gap: 12,
    marginBottom: 0,
  },
  reportTypeIcon: {
    padding: 10,
    borderRadius: 10,
  },
  reportTypeText: {
    flex: 1,
  },
  reportTypeTitle: {
    color: '#ffffff',
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 4,
  },
  reportTypeDesc: {
    color: '#94a3b8',
    fontSize: 13,
    lineHeight: 18,
  },

  // CTA card
  cta: {
    alignItems: 'center',
    paddingVertical: 24,
    marginBottom: 16,
  },
  ctaIcon: {
    fontSize: 36,
    marginBottom: 8,
  },
  ctaTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  ctaDesc: {
    fontSize: 13,
    color: '#94a3b8',
    textAlign: 'center',
    marginBottom: 20,
    lineHeight: 20,
  },
  reportBtn: {
    backgroundColor: '#6366f1',
    paddingVertical: 13,
    paddingHorizontal: 32,
    borderRadius: 8,
    width: '100%',
    alignItems: 'center',
    marginBottom: 12,
  },
  reportBtnText: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  anonymousBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(74,222,128,0.1)',
    borderWidth: 1,
    borderColor: 'rgba(74,222,128,0.2)',
    paddingVertical: 7,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  anonymousText: {
    color: '#4ade80',
    fontSize: 13,
    fontWeight: '600',
  },

  // Emergency
  emergencyCard: {
    backgroundColor: 'rgba(0,0,0,0.35)',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.05)',
    marginBottom: 8,
  },
  emergencyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 8,
  },
  emergencyDesc: {
    fontSize: 13,
    color: '#94a3b8',
    textAlign: 'center',
    marginBottom: 16,
    lineHeight: 20,
  },
  callBtn: {
    backgroundColor: '#ef4444',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
  },
  callBtnText: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
